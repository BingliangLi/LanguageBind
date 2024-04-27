import os
import torch
import numpy as np
import json
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from multiprocessing import Process, Manager
from tqdm import tqdm
import time

def process_audio(audio_files, input_folder, output_folder, device_id, failed_audios):
    # Set device
    device = torch.device(f'cuda:{device_id}')

    # Model setup, using the correct clip_type for audio
    clip_type = {'audio': 'LanguageBind_Audio_FT'}
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()

    # Model tokenizer and transformation (if needed)
    # Note: Tokenizer might not be needed for audio, depending on model requirements
    modality_transform = transform_dict['audio'](model.modality_config['audio'])

    # Process the assigned audio files with a progress bar
    pbar = tqdm(audio_files, desc=f"GPU-{device_id}", position=device_id)
    for filename in pbar:
        audio_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_audio_embedding.npy")

        # Check if the feature file already exists
        if os.path.exists(output_path):
            continue
        try:
            # Prepare audio input
            audio_input = to_device(modality_transform(audio_path), device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = model({'audio': audio_input})

            # Save embeddings to .npy format
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_audio_embedding.npy")
            np.save(output_path, embeddings['audio'].cpu().numpy())
            pbar.set_description(f"GPU-{device_id} processed {filename}")
            
            time.sleep(2)  # Throttle the CPU usage

        except Exception as e:
            pbar.set_description(f"GPU-{device_id} error on {filename}")
            failed_audios.append(audio_path)

def split_processing(input_folder, output_folder, num_gpus=8):
    manager = Manager()
    failed_audios = manager.list()

    # Get all audio files
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.flac')]
    num_files = len(audio_files)
    part = num_files // num_gpus

    processes = []
    for i in range(num_gpus):
        start_index = i * part
        end_index = start_index + part if i != num_gpus - 1 else num_files
        subset_files = audio_files[start_index:end_index]
        process = Process(target=process_audio, args=(subset_files, input_folder, output_folder, i, failed_audios))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

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
