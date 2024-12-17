## launch docker
```` shell
bash run_docker.sh
````
## training scripts

```` shell
# train llama2
bash train_cp_mock.sh MODEL_SIZE=7 TP=1 PP=1 MBS=7 BS=280 TE_FP8=1 SEQ_LENGTH=4096
bash train_cp_mock.sh MODEL_SIZE=70 TP=8 PP=1 MBS=4 BS=256 TE_FP8=1 SEQ_LENGTH=4096
````
