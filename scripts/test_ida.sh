#!/bin/bash

#SBATCH --job-name=RePRI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=lg154@nyu.edu
#SBATCH --output=seg.out
#SBATCH --gres=gpu:1  # How much gpu need, n is the number

DATA=$1
SHOT=$2
GPU=$3
LAYERS=$4

SPLITS="0 1 2 3"

if [ $SHOT == 1 ]
then
   bsz_val="200"
elif [ $SHOT == 5 ]
then
   bsz_val="100"
elif [ $SHOT == 10 ]
then
   bsz_val="50"
fi

for SPLIT in $SPLITS
do
  dirname="results/test/arch=resnet-${LAYERS}/data=${DATA}/shot=shot_${SHOT}/split=split_${SPLIT}"
	mkdir -p -- "$dirname"
  echo "start"
  singularity exec --nv \
              --overlay /scratch/lg154/python36/python36.ext3:ro \
              --overlay /scratch/lg154/sseg/dataset/coco2014.sqf:ro \
              /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
              /bin/bash -c " source /ext3/env.sh;
              python -m src.test_ida --config config_files/${DATA}.yaml \
               --opts train_split ${SPLIT} \
                  batch_size_val 1 \
							   shot ${SHOT} \
							   layers ${LAYERS} \
							   FB_param_update "[10]" \
							   temperature 20.0 \
							   adapt_iter 50 \
							   cls_lr 0.025 \
							   gpus ${GPU} \
							   test_num 1000 \
							   n_runs 1 \
             > ${dirname}/log_test_ida.txt 2>&1"

  echo "finish"


done