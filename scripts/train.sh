#!/bin/bash

#SBATCH --job-name=seg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu # How much gpu need, n is the number

module purge

DATA=$1
SPLIT=$2
GPU=$3
LAYERS=$4

dirname="results/train/resnet-${LAYERS}/${DATA}/split_${SPLIT}"
mkdir -p -- "$dirname"

echo "start"
singularity exec --nv \
            --overlay /scratch/lg154/python36/python36.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c " source /ext3/env.sh;
            python -m src.train --config config_files/${DATA}.yaml \
					  --opts train_split ${SPLIT} \
						    layers ${LAYERS} \
						    gpus ${GPU} \
						    visdom_port 8098 \
							 > ${dirname}/log_${SHOT}.txt 2>&1"

echo "finish"


#GREENE GREENE_GPU_MPS=yes