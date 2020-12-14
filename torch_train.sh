#!/bin/bash

# usage
# ./torch_train.sh 2 nv2080ti-test

cat >"./.sbatch.tmp" <<-EOM
set -x
#SBATCH --nodes=$1
#SBATCH --gres=gpu:8
#SBATCH --partition=$2
#SBATCH --workdir=$(pwd)
#SBATCH --framework=artifactory.momenta.works/docker-momenta/lane_heat_pytorch:v0.0.10
#SBATCH --dataset=234huan,lidar-open-dataset
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0

HOSTADDR=\$(head -n 1 /tmp/.mpirun.hostfile | awk '{print \$1}')
export HOSTADDR

# must set --bind-to none for apex, see https://stackoverflow.com/a/25772686/7636233
# otherwise will only use one core per numa node

# mpirun -npernode 1 -np $1 \
#   -x NCCL_DEBUG -x NCCL_TREE_THRESHOLD -x NCCL_SOCKET_IFNAME \
#   -x LD_LIBRARY_PATH -x PATH \
#   -x HOSTADDR  --bind-to none \
#   bash -c 'python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=$1 \
#     --node_rank=\$OMPI_COMM_WORLD_RANK --master_addr=\$HOSTADDR \
#     train_imagenet.py --dist torch --log-dir=./torch_log/' | tee torch_run.log

python loop.py
EOM

jobname='jingwei'$(date '+%Y%m%d%H%M%S')

slurm batch ./.sbatch.tmp -J $jobname
