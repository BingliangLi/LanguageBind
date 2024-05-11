import os
import json
import torch
import numpy as np
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
from tqdm import tqdm

def load_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def process_text_descriptions(data, model, tokenizer, device, output_folder):
    for audio_id, caption in tqdm(data.items()):
        try:
            # Prepare text input
            text_input = to_device(tokenizer([caption], max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)
            # Generate embeddings
            with torch.no_grad():
                embeddings = model({'language': text_input})

            # Save embeddings to .npy format
            embedding_path = os.path.join(output_folder, f"{audio_id}.npy")
            np.save(embedding_path, embeddings['language'].cpu().numpy())

            # Save attention mask to .npy format
            attention_mask_path = os.path.join(output_folder, f"{audio_id}_attention_mask.npy")
            attention_mask = text_input['attention_mask'].cpu().numpy()
            np.save(attention_mask_path, attention_mask)

            print(f"Saved embeddings and attention mask for {audio_id} to {output_folder}")

        except Exception as e:
            print(f"Error processing {audio_id}: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_text_embeddings_from_json.py <json_path> <output_folder>")
        sys.exit(1)

    json_path = sys.argv[1]
    output_folder = sys.argv[2]

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Set device and initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_type = {'language': 'LanguageBind_Image'}
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
