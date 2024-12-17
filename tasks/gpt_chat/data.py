import glob
import json
import math
import os
import time
import copy
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.training import print_rank_0
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_ids
from tasks.data_utils import clean_text
from tasks.gpt_chat.data_utils import preprocess, PREFIX_STR


logger = logging.getLogger(__file__)


class ChatDataset(Dataset):

    def __init__(self, dataset_name, datapaths, tokenizer,
            max_seq_length: int,
            pad_to_max_length: bool = False,
            tokens_to_generate: int = 0,
            ceil_to_power_2: bool = False,
            ):

        # Init tokenizer
        self.tokenizer = tokenizer
        special_tokens = {
            "system_turn_start": "<extra_id_0>",
            "turn_start": "<extra_id_1>",
            "label_start": "<extra_id_2>",
            "end_of_turn": "\n",
            "end_of_name": "\n",
        }
        self.special_tokens = special_tokens

        LABEL_START = self.special_tokens['label_start']
        END_NAME_SIGNAL = self.special_tokens['end_of_name']

        id1 = self.tokenize(PREFIX_STR)
        id2 = self.tokenize(PREFIX_STR + LABEL_START)
        self.label_start_tokens = id2[len(id1) :]

        id1 = self.tokenize(PREFIX_STR + END_NAME_SIGNAL)
        id2 = self.tokenize(PREFIX_STR)
        self.name_end_token_ids = id1[len(id2) :]

        id1 = self.tokenize(PREFIX_STR + self.special_tokens['turn_start'])
        id2 = self.tokenize(PREFIX_STR)
        self.num_turn_start_tokens = len(id1) - len(id2)
        self.turn_start_tokens = id1[len(id2) :]

        self.dataset_name = dataset_name
        print_rank_0(' > building chat dataset for {}:'.format(
            self.dataset_name))

        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        print_rank_0(string)

        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length
        print_rank_0(' > max sequence length {}:'.format(
            self.max_seq_length))
        self.tokens_to_generate = tokens_to_generate
        self.ceil_to_power_2 = ceil_to_power_2

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(self.process_single_datapath(datapath))

        print_rank_0('  >> total number of samples: {}'.format(
            len(self.samples)))

    def tokenize(self, text):
        return self.tokenizer._tokenizer.convert_tokens_to_ids(self.tokenizer._tokenizer.tokenize(text))

    def process_single_datapath(self, datapath):
        """Read in RACE files, combine, clean-up, tokenize, and convert to
        samples."""

        print_rank_0('   > working on {}'.format(datapath))
        start_time = time.time()

        samples = []

        fin = open(datapath, 'r', encoding='utf-8')
        for jsonl in fin:
            data = json.loads(jsonl)
            samples.append(data)

        elapsed_time = time.time() - start_time
        print_rank_0('    > processed {} samples'
                    ' in {:.2f} seconds'.format(len(samples), elapsed_time))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)
        result = preprocess(data, self.tokenizer, self.name_end_token_ids, self.label_start_tokens, self.special_tokens, self.num_turn_start_tokens)
        result["metadata"] = {}

        return result

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _ceil_to_nearest(self, n, m):
        if self.ceil_to_power_2:
            # Reccurent Gemma (AKA Griffin) requires seq length to be a power of 2 for parallel scan
            return 2 ** math.ceil(math.log2(n))
        else:
            return (n + m - 1) // m * m

    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item


    @torch.no_grad()
    def _create_attention_mask(self, max_length):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def _collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1].tolist() for item in batch]
        labels = [item['input_ids'][1:].tolist() for item in batch]
        contexts = [item['context_ids'].tolist() for item in batch]
        answers = [item['answer_ids'].tolist() for item in batch]
        loss_mask = [item['mask'][1:].tolist() for item in batch]
        metadata = [item['metadata'] for item in batch]

        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)
        if max_length > self.max_seq_length:
            # truncate the sequences if it is longer than max_seq_length
            input_ids = [x[: self.max_seq_length] for x in input_ids]
            labels = [x[: self.max_seq_length] for x in labels]
            loss_mask = [x[: self.max_seq_length] for x in loss_mask]
            contexts = [x[: self.max_seq_length] for x in contexts]
            answers = [x[: self.max_seq_length] for x in answers]

        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length

        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer._tokenizer.eos_token_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer._tokenizer.eos_token_id))
        loss_mask = torch.FloatTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        context_lengths = torch.LongTensor([len(x) for x in contexts])
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer._tokenizer.eos_token_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer._tokenizer.eos_token_id))

        processed_batch = {
            'tokens': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts,
            'context_lengths': context_lengths,
            'answers': answers,
            'metadata': metadata,
        }

        return processed_batch

