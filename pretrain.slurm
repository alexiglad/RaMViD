#!/bin/bash -l

#SBATCH -A uvadm
#SBATCH --array=0
#SBATCH -t 72:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:3
#SBATCH --constraint=a100_80gb
#SBATCH -N 1
#SBATCH --cpus-per-task=40
#SBATCH --output=logs/slurm/log-%A-%a.log
#SBATCH -J ramvid

MODEL_FLAGS="--image_size 224 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 4 --max_num_mask_frames 3 --uncondition_rate 0.25"
#TODO change code so no longer pretrains on test set
export OPENAI_LOGDIR=./logs

singularity exec --env PYTHONPATH=/opt/conda/lib/python3.10/site-packages --nv /scratch/abg4br/containers/ramvid_env.sif mpirun --oversubscribe --host localhost:3 -n 3 python scripts/video_train.py --data_dir ../something-something-v2/videos/ $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS


