NAME=training_llama2
DOCKER=rocm/pytorch-private:exec_dash_pretuned_nightly_inai_FA_ck_v0.1.1_bf16_rtn_TE_GQA
docker run -it --name $NAME --device /dev/kfd --device /dev/dri \
    --privileged --network=host \
    --group-add render --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --shm-size=64g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${PWD}:/workspace \
    -v /mnt/md0:/mnt/md0 \
    -v /home:/home \
    -w /workspace \
    $DOCKER \
    /bin/bash
