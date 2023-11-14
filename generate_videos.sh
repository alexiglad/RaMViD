MODEL_FLAGS="--image_size 224 --num_channels 128 --num_res_blocks 3 --scale_time_dim 0"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
# TRAIN_FLAGS="--lr 2e-5 --batch_size 8 --microbatch 2 --seq_len 4 --max_num_mask_frames 3 --uncondition_rate 0.25"

INFERENCE_FLAGS="--timestep_respacing 500 --cond_frames 0,1,2, --seq_len 19 --save_gt True --batch_size 10 --num_samples 120 --data_dir ../ssv2_just_test/" 
export OPENAI_LOGDIR=./logs
# --data_dir ../something-something-v2/videos/ NOTE old
singularity exec --env PYTHONPATH=/opt/conda/lib/python3.10/site-packages --nv /scratch/abg4br/containers/ramvid_env.sif mpirun --oversubscribe --host localhost:3 -n 3 python scripts/video_sample.py --model_path /scratch/abg4br/RaMViD/logs/ema_0.9999_114000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $INFERENCE_FLAGS