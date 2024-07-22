import os
import torch
import numpy as np
import json
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from multiprocessing import Process, Manager
from tqdm import tqdm
import time

def process_videos(video_files, input_folder, output_folder, device_id, failed_videos):
    # Set device
    device = torch.device(f'cuda:{device_id}')

    # Model setup, ensure clip_type is a dictionary as expected by the LanguageBind model
    clip_type = {'video': 'LanguageBind_Video_V1.5_FT'}
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()

    # Model tokenizer and transformation
    pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = transform_dict['video'](model.modality_config['video'])

    # Process the assigned video files with a progress bar
    pbar = tqdm(video_files, desc=f"GPU-{device_id}", position=device_id)
    for filename in pbar:
        video_path = os.path.join(input_folder, filename)
        try:
            # Prepare video input
            video_input = to_device(modality_transform(video_path), device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = model({'video': video_input})

            # Save embeddings to .npy format
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_video_embedding.npy")
            np.save(output_path, embeddings['video'].cpu().numpy())
            pbar.set_description(f"GPU-{device_id} processed {filename}")

        except Exception as e:
            pbar.set_description(f"GPU-{device_id} error on {filename}")
            failed_videos.append(video_path)
        time.sleep(0.5)

def split_processing(input_folder, output_folder, num_gpus=8):
    manager = Manager()
    failed_videos = manager.list()

    # Get all video files
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]
    num_files = len(video_files)
    part = num_files // num_gpus

    processes = []
    for i in range(num_gpus):
        start_index = i * part
        end_index = start_index + part if i != num_gpus - 1 else num_files
        subset_files = video_files[start_index:end_index]
        process = Process(target=process_videos, args=(subset_files, input_folder, output_folder, i, failed_videos))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    # After all processes complete, write failed videos to JSON
    failed_videos_file = os.path.join(output_folder, f"{os.path.basename(os.path.normpath(input_folder))}_failed_videos.json")
    with open(failed_videos_file, 'w') as f:
        json.dump(list(failed_videos), f)
    print(f"Failed video paths saved to {failed_videos_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_video_embeddings.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process the videos in parallel
    split_processing(input_folder, output_folder)
