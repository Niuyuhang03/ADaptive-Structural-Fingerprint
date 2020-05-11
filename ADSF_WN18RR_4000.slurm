#!/bin/bash
#SBATCH -o ./ADSF/ADSF_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J ad_w4             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
time=`date +%Y%m%d-%H%M%S`
mkdir ./ADSF/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py --no-sparse --dataset WN18RR_4000 --experiment ADSF/${time}
mv ./ADSF/ADSF_result_$SLURM_JOB_ID.log ./ADSF/${time}/ADSF_result.log
