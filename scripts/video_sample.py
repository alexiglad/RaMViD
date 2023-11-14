"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import time
from datetime import datetime
from torchvision.transforms import ToPILImage
import torch
import numpy as np
import torch as th
import torch.distributed as dist

import os, sys
sys.path.insert(1, os.getcwd()) 
import random

from diffusion_openai.video_datasets import load_data
from diffusion_openai import dist_util, logger
from diffusion_openai.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    if args.seed:
        th.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    cond_kwargs = {}
    cond_frames = []
    if args.cond_generation:
        data = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=args.class_cond,
            deterministic=True,
            rgb=args.rgb,
            seq_len=args.seq_len
        )
        num = ""
        for i in args.cond_frames:
            if i == ",":
                cond_frames.append(int(num))
                num = ""
            else:
                num = num + i
        ref_frames = list(i for i in range(args.seq_len) if i not in cond_frames)
        logger.log(f"cond_frames: {cond_frames}")
        logger.log(f"ref_frames: {ref_frames}")
        logger.log(f"seq_len: {args.seq_len}")
        cond_kwargs["resampling_steps"] = args.resample_steps
    cond_kwargs["cond_frames"] = cond_frames

    if args.rgb:
        channels = 3
    else:
        channels = 1

    logger.log("sampling...")
    all_videos = []
    all_gt = []

    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    dt_string = dt_object.strftime("%Y-%m-%d_%H-%M-%S")
    num_generated_videos = 0
    dataset_save_base_dir = "../generated_datasets/ramvid"
    save_video_dir = f"{dataset_save_base_dir}/{dt_string}"
    print("about to start while loop dist.get_rank()", dist.get_rank())
    #TODO change code below to all gather and only do if is master process
    while num_generated_videos < args.num_samples:
        
        if args.cond_generation:
            video, _ = next(data)
            cond_kwargs["cond_img"] = video[:,:,cond_frames].to(dist_util.dev()) 
            video = video.to(dist_util.dev())


        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        sample = sample_fn(
            model,
            (args.batch_size, channels, args.seq_len, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            progress=True,
            cond_kwargs=cond_kwargs
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 1, 3, 4) # also had to switch last two dims images were sideways
        #after this is 10, 19, 224, 224, 3
        #before it was 10, 3, 19, 224, 224
        #so should do
        # sample.permute(0, 2, 1, 4, 3 (w and h switched here see above)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_samples = torch.cat(gathered_samples, dim=0)
        # all_videos.extend([sample.cpu().numpy() for sample in gathered_samples])
        if dist.get_rank() == 0:
            save_videos_as_frames(gathered_samples, save_video_dir, 'fake', num_generated_videos)
        

        if args.cond_generation and args.save_gt:

            video = ((video + 1) * 127.5).clamp(0, 255).to(th.uint8)
            video = video.permute(0, 2, 1, 3, 4)
            video = video.contiguous()
            gathered_videos = [th.zeros_like(video) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_videos, video)  # gather not supported with NCCL
            gathered_videos = torch.cat(gathered_videos, dim=0)
            # all_gt.extend([video.cpu().numpy() for video in gathered_videos])
            # logger.log(f"created {len(all_gt) * args.batch_size} videos")
            if dist.get_rank() == 0:
                save_videos_as_frames(gathered_videos, save_video_dir, 'real', num_generated_videos)
            


        num_generated_videos += len(sample) * dist.get_world_size()
        logger.log(f"created {num_generated_videos} samples so far")


    # arr = np.concatenate(all_videos, axis=0)

    # if args.cond_generation and args.save_gt:
    #     arr_gt = np.concatenate(all_gt, axis=0)
    # NOTE commented out all this code as saving all vids in online manner

    # if dist.get_rank() == 0:

    #     shape_str = "pred" + "x".join([str(x) for x in arr.shape])
    #     logger.log(f"saving samples to {os.path.join(logger.get_dir(), shape_str)}")
    #     np.savez(os.path.join(logger.get_dir(), shape_str), arr)

    #     if args.cond_generation and args.save_gt:
    #         shape_str_gt = "gt" + "x".join([str(x) for x in arr_gt.shape])
    #         logger.log(f"saving ground_truth to {os.path.join(logger.get_dir(), shape_str_gt)}")
    #         np.savez(os.path.join(logger.get_dir(), shape_str_gt), arr_gt)


    dist.barrier()
    logger.log("sampling complete")

def save_videos_as_frames(video_list, root_dir, subfolder, start_index=0):
    video_list = video_list[:, 3:]#condition on first 3 frames so dont use
    to_pil = ToPILImage()
    subfolder_path = os.path.join(root_dir, subfolder)
    os.makedirs(subfolder_path, exist_ok=True)
    
    # Iterate over each video in the list
    for video_idx, video in enumerate(video_list):
        # Create a directory for the current video with a unique index
        video_dir = os.path.join(subfolder_path, f'video_{start_index + video_idx}')
        os.makedirs(video_dir, exist_ok=True)

        # Iterate over each frame in the video
        for frame_idx, frame in enumerate(video):
            # Convert the frame to a PIL image
            frame_img = to_pil(frame.cpu().detach())

            # Save the frame image with the appropriate filename
            frame_img.save(os.path.join(video_dir, f'frame_{frame_idx}.png'))


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1, #TODO change code to produce whole ds of samples later
        batch_size=10,
        use_ddim=False,
        model_path="",
        seq_len=16,
        sampling_type="generation",
        cond_frames="0,",
        cond_generation=True,
        resample_steps=1,
        data_dir='',
        save_gt=False,
        seed = 0
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print(f"elapsed time: {end - start}")