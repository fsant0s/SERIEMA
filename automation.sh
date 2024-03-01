docker run --gpus '"device=0,1,2"' -ti \
    -e OUTSIDE_USER=$USER \
    -e OUTSIDE_UID=$UID \
    -e OUTSIDE_GROUP='/usr/bin/id' \
    -e OUTSIDE_GID='/usr/bin/id' \
    --userns=host \
    --shm-size 24G \
    -v /work/fillipe.silva/SIRIEMA/:/SIRIEMA/ \
    -v /hadatasets/fillipe.silva/processed/:/SIRIEMA/data/ \
    -p 2020:2020 \
    --name siriema_container \
    siriema:latest /bin/bash

# Additional commands
echo "To start the environment: docker start $DOCKER_CONTAINER_NAME"
echo "To stop the environment: docker stop $DOCKER_CONTAINER_NAME"
echo "To remove the environment: docker rm $DOCKER_CONTAINER_NAME"
echo "To attach to the environment: docker attach $DOCKER_CONTAINER_NAME"

#CUDA_VISIBLE_DEVICES=0 python py.py
#watch -n 1 nvidia-smi
#ssh -L 2020:dl-08:2020 recodssh