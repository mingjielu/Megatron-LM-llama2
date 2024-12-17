import glob
import json
import os
import time
import copy
import logging

import torch
from torch.utils.data import Dataset

from megatron.training import print_rank_0
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_ids
from tasks.data_utils import clean_text


logger = logging.getLogger(__file__)


PREFIX_STR = (
    "\x00"  # the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
)

IGNORE_INDEX = -100
SYSTEM_TOKEN = "System"

TYPE_INSTRUCTION = {
    'TEXT_TO_VALUE': "",
    'VALUE_TO_TEXT': '',
}

def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    data_type = None
    if 'type' in source:
        data_type = source['type']
        if data_type is not None:
            assert data_type in TYPE_INSTRUCTION, f"source type {data_type} not supported"
    # add end signal and concatenate together
    conversation = source['system']
    if data_type is not None:
        if TYPE_INSTRUCTION[data_type] != '':
            conversation = conversation + '\n' + TYPE_INSTRUCTION[data_type]
    mask_role = source.get('mask', 'User')
    header = f"{special_tokens['system_turn_start']}{SYSTEM_TOKEN}{END_NAME_SIGNAL}{conversation}{END_SIGNAL}"
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role

def response_value_formater(label, label_start, end_signal):
    if isinstance(label, str):
        return label_start + label + end_signal
    elif label is None:
        return ''
    else:
        raise ValueError(f'Unknown label type {type(label)}, only str type is supported')

def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    LABEL_START = special_tokens['label_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']

    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = ""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        role_token = TURN_TOKEN
        if gtype is None:
            sentence["value"] = (
                BEGIN_SIGNAL + role_token + sentence_from + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
            )
        elif gtype == "VALUE_TO_TEXT":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
                + sentence["value"]
                + END_SIGNAL
            )
        elif gtype == "TEXT_TO_VALUE":
            sentence["value"] = (
                BEGIN_SIGNAL
                + role_token
                + sentence_from
                + END_NAME_SIGNAL
                + sentence["value"]
                + END_SIGNAL
                + (
                    response_value_formater(sentence['label'], LABEL_START, END_NAME_SIGNAL)
                    if 'label' in sentence
                    else ''
                )
            )
        else:
            raise ValueError(
                f"source type {gtype} not supported, only 'VALUE_TO_TEXT' and 'TEXT_TO_VALUE' are supported"
            )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation

def identify_start_index_of_subsequence(subsequence, sequence):
    """ find the location of the small tensor in the large tensor.
        e.g.  small = [1,3], large = [2,3,1,3], returns 2
              small = [3,2], large = [2,3,1,3], returns -1
    Args:
        small (tensor): small tensor
        large (tensor): large tensor
    """
    for i in range(sequence.size(0) - subsequence.size(0) + 1):
        if torch.equal(sequence[i : i + subsequence.size(0)], subsequence):
            return i
    return -1

def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """ This function masks the tokens so the loss is computed only on the non-masked role's responses.
    For 'TEXT_TO_VALUE' type, the loss is computed on the value attributes.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """
    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    label_start_ids = torch.tensor(label_start_ids)
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(PREFIX_STR))
        id2 = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(PREFIX_STR + TURN_TOKEN + speaker + END_NAME_SIGNAL))
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'
        # get the position of the label start string in this turn
        location = identify_start_index_of_subsequence(label_start_ids, s_id)

        if location >= 0:
            # if it contains the label start tokens
            if gtype == 'VALUE_TO_TEXT':
                # handles the case that condition on labels to generate respone
                # the next token after the name part of the prompt is the beginning of the label start tokens
                assert skip_name_len == location
                # find the first new line token after the label part, which indicates the end of the whole label string
                # newline_loc = torch.where((s_id[skip_name_len:] == name_end_token_ids))[0]
                newline_loc = identify_start_index_of_subsequence(name_end_token_ids, s_id[skip_name_len:])
                if newline_loc < 0:
                    # cannot find new line token, which means the the whole turn is just a partial label string. Mask the whole turn
                    target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
                    continue
                # skip the label part and the new line token
                more_skip_len = newline_loc + len(name_end_token_ids)
                # skip the name part and the label part
                skip_name_len += more_skip_len
            elif gtype == 'TEXT_TO_VALUE':
                # handles the case that condition on response to generate label
                # skip the name part, response and the label start tokens part, the remainder is the label string without label start, e.g. 'quality:9,toxicity:8...'
                skip_name_len = location + len(label_start_ids)
        if cur_idx >= tgt_len:
            break
        elif cur_idx + tokenized_len < tgt_len:
            # Check whether the mask is applied to the correct position, the first token is turn start tokens
            if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
                logger.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0 and (gtype == 'VALUE_TO_TEXT' or gtype is None):
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
            # leave the first turn start tag unmasked, servers severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i <= 1):
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask up to name part, label part for VALUE_TO_TEXT, or name part, response and label start tokens for TEXT_TO_VALUE, or just the name part if gtype is None
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def preprocess(
    source: dict,
    tokenizer,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    # tokenize conversations
    input_ids = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(conversation))
    target = copy.deepcopy(input_ids)
    header_tokens = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(header))
    header_len = len(header_tokens)

    ids = []
    tokenized_lens = []
    assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(PREFIX_STR + s["value"]))
        id2 = tokenizer._tokenizer.convert_tokens_to_ids(tokenizer._tokenizer.tokenize(PREFIX_STR))
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    target = torch.LongTensor(target)
    # not going to train on the header
    target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )
    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]
    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)
