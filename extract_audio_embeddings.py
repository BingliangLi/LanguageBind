import os
import torch
import numpy as np
import json
from tqdm import tqdm
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from multiprocessing import Pool, Manager, cpu_count
import time

def process_audio(args):
    filename, input_folder, output_folder, device_id, failed_audios = args
    # Initialize device and model for each worker
    device = torch.device(f'cuda:{device_id}')
    clip_type = {'audio': 'LanguageBind_Audio_FT'}
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    modality_transform = transform_dict['audio'](model.modality_config['audio'])

    audio_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_audio_embedding.npy")
    
    # Check if the output file already exists
    if os.path.exists(output_path):
        return f"Already exists: {filename}"

    try:
        # Prepare audio input
        audio_input = to_device(modality_transform(audio_path), device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = model({'audio': audio_input})

        # Save embeddings to .npy format
        np.save(output_path, embeddings['audio'].cpu().numpy())
        # Artificially throttle CPU usage
        time.sleep(0.1)
        return f"Processed: {filename}"

    except Exception as e:
        failed_audios.append(audio_path)
        return f"Error on {filename}: {e}"

def split_processing(input_folder, output_folder, num_gpus=8):
    manager = Manager()
    failed_audios = manager.list()

    # Get all audio files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.flac')]
    
    # Preparing arguments for multiprocessing
    tasks = [(filename, input_folder, output_folder, i % num_gpus, failed_audios) for i, filename in enumerate(audio_files)]
    
    # Calculate the number of processes to limit CPU usage
    num_processes = max(1, int(cpu_count() * 0.4))  # Use only 40% of CPU cores
    
    # Using Pool to manage parallel processing
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_audio, tasks), total=len(tasks), desc="Processing Audios"):
            print(result)

    # After all processes complete, write failed audios to JSON
    failed_audios_file = os.path.join(output_folder, f"{os.path.basename(os.path.normpath(input_folder))}_failed_audios.json")
    with open(failed_audios_file, 'w') as f:
        json.dump(list(failed_audios), f)
    print(f"Failed audio paths saved to {failed_audios_file}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_audio_embeddings.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process the audios in parallel
    split_processing(input_folder, output_folder)
