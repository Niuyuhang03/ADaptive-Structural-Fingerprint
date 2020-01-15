#!/bin/bash
#SBATCH -o ADFS.log         # 输出日志，可以查看ckpt文件夹
#SBATCH -J ADFS             # 任务名称，只能显示前6字符
#SBATCH --gres=gpu:V100:1   # 申请1个v100的gpu
#SBATCH -c 5        # 申请5个cpu核心
#SBATCH -p sugon	# 指定使用曙光gpu，参数为cpu则在cpu上跑
#SBATCH --mail-user=user@mail.com	# 作业结束后发送邮件

source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate cuda9.1
CUDA_VISIBLE_DEVICES=0 python train.py
