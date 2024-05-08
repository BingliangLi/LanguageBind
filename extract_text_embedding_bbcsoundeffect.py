import os
import json
import torch
import numpy as np
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from tqdm import tqdm

def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['data']

def process_text_descriptions(data, model, tokenizer, device, output_folder):
    for item in tqdm(data):
        audio_id = item['id']
        audio_file = f"{audio_id}.flac"
        caption = item['caption']

        if os.path.exists(audio_file):
            try:
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
                print(f"Saved embeddings for {audio_id} to {output_path}")

            except Exception as e:
                print(f"Error processing {audio_id}: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print("Usage: python extract_bbc_sound_effect_text_embeddings.py <audio_dir> <json_path> <output_folder>")
        sys.exit(1)

    audio_dir = sys.argv[1]
    json_path = sys.argv[2]
    output_folder = sys.argv[3]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set device and initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_type = {'image': 'LanguageBind_Image'}  # Assuming text processing is using a 'LanguageBind_Image' or similar
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()

    # Load tokenizer
    pretrained_ckpt = 'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')

    # Load data from JSON
    data = load_data(json_path)

    # Process text descriptions
    process_text_descriptions(data, model, tokenizer, device, output_folder)



# if __name__ == '__main__':
#     audio_dir = '/workspace/mnt/mm_data/libingliang/BBC_new/audio_441_stereo'
#     json_path = '/workspace/mnt/mm_data/libingliang/BBC_new/bbc_final.json'
#     output_dir = '/workspace/mnt/mm_data/libingliang/BBC_new/text_embedding'
    
#     extract_bbc_sound_effect_text_embeddings(audio_dir, json_path, output_dir)
