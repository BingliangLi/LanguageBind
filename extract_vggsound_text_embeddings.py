import os
import csv
import torch
import numpy as np
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from tqdm import tqdm
from multiprocessing import Process

failed_audios = []
missing_audio_files = []

global failed_audios
global missing_audio_files

def load_data(csv_file):
    data = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append({
                'id': row[0],
                'caption': row[1],
            })
    return data

def get_random_list_element(list):
    return list[np.random.randint(0, len(list))]

def caption_augmentation(caption):
    prefix = ["The sound of ", "A sound of ", "The audio of ", "A recording of ", "A clip of "]
    caption = get_random_list_element(prefix) + caption
    return caption

def process_text_descriptions(data, model, tokenizer, device, audio_dir, output_folder):
    print_example = True
    for item in tqdm(data, desc=f"Process on GPU {device}"):
        audio_id = item['id']
        # remove '.mp4'
        audio_id = audio_id[:-4]
        audio_file = f"{audio_id}.flac"
        audio_path = os.path.join(audio_dir, audio_file)
        caption = item['caption']
        caption = caption_augmentation(caption)
        if print_example:
            print(f"Example caption: {caption}")
            print_example = False

        if os.path.exists(audio_path):
            try:
                # Set device for the model and data
                torch.cuda.set_device(device)
                model.to(device)

                # Prepare text input
                text_input = to_device(tokenizer([caption], max_length=77, padding='max_length',
                                                 truncation=True, return_tensors='pt'), device)
                # Generate embeddings
                with torch.no_grad():
                    embeddings = model({'language': text_input})

                # Save embeddings to .npy format
                output_path = os.path.join(output_folder, f"{audio_id}.npy")
                np.save(output_path, embeddings['language'].cpu().numpy())
                # Save attention mask to .npy format
                attention_mask = text_input['attention_mask'].cpu().numpy()
                np.save(os.path.join(output_folder, f"{audio_id}_attention_mask.npy"), attention_mask)

            except Exception as e:
                failed_audios.append(audio_path)
                print(f"Error processing {audio_id} on GPU {device}: {e}")
        else:
            missing_audio_files.append(audio_path)
            print(f"Audio file not found: {audio_path}")
        
    print("number of failed audios: ", len(failed_audios))
    print("number of missing audio files: ", len(missing_audio_files))

def split_and_process(data, audio_dir, output_folder, num_gpus=8):
    # Split data into parts for each GPU
    part_size = len(data) // num_gpus
    processes = []

    for i in range(num_gpus):
        start = i * part_size
        end = start + part_size if i != num_gpus - 1 else len(data)
        subset = data[start:end]

        # Initialize model and tokenizer per process
        device = f'cuda:{i}'
        clip_type = {'audio': 'LanguageBind_Audio_FT'}
        model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
        model.eval()
        pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
        tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')

        # Create a process
        p = Process(target=process_text_descriptions, args=(subset, model, tokenizer, device, audio_dir, output_folder))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python process_vggsound_text_embeddings.py <audio_dir> <csv_path> <output_folder>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    csv_path = sys.argv[2]
    output_folder = sys.argv[3]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load data from CSV
    data = load_data(csv_path)

    # Process data in parallel
    split_and_process(data, audio_dir, output_folder)
    
    # save failed_audios as json
    failed_audios_file = os.path.join(output_folder, f"failed_audios.json")
    with open(failed_audios_file, 'w') as f:
        f.write(str(failed_audios))
        
    # save missing_audio_files as json
    missing_audio_files_file = os.path.join(output_folder, f"missing_audio_files.json")
    with open(missing_audio_files_file, 'w') as f:
        f.write(str(missing_audio_files))
    
    print("number of failed audios: ", len(failed_audios))
    print(failed_audios_file)
    print("number of missing audio files: ", len(missing_audio_files))
    print(missing_audio_files_file)
