#!/bin/bash
#SBATCH -o ./RWR/RWR_result_%j.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J rwr_w4             # 任务名称
#SBATCH --gres=gpu:V100:1               # 申请一张GPUV100
#SBATCH -c 5                # 申请CPU核数
#SBATCH -p sugon
time=`date +%Y%m%d-%H%M%S`
mkdir ./RWR/${time}
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 1  --dataset WN18RR_4000 --experiment RWR/${time}
mv ./RWR/RWR_result_$SLURM_JOB_ID.log ./RWR/${time}/RWR_result.log
