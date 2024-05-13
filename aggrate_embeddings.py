import os
import numpy as np
import pickle
import sys
from tqdm import tqdm

def load_embedding(file_path):
    try:
        return np.load(file_path)
    except:
        return None

def main(audio_dir, text_emb_dir, audio_emb_dir, video_emb_dir, save_path):
    # Dictionary to hold all embeddings
    multi_model_embeddings = {}

    # Get all file_ids from the audio directory
    if os.path.isdir(audio_dir):
        file_ids = [file.split('.')[0] for file in os.listdir(audio_dir) if file.endswith('.flac')]
    else:
        print("The audio directory is not valid.")
        return

    # Process each file_id to gather embeddings
    for file_id in tqdm(file_ids):
        text_embedding = None
        audio_embedding = None
        video_embedding = None

        if text_emb_dir and os.path.isdir(text_emb_dir):
            text_embedding = load_embedding(os.path.join(text_emb_dir, f"{file_id}.npy"))
        
        if audio_emb_dir and os.path.isdir(audio_emb_dir):
            audio_embedding = load_embedding(os.path.join(audio_emb_dir, f"{file_id}_audio_embedding.npy"))
        
        if video_emb_dir and os.path.isdir(video_emb_dir):
            video_embedding = load_embedding(os.path.join(video_emb_dir, f"{file_id}_video_embedding.npy"))
        
        multi_model_embeddings[file_id] = {
            "text_embedding": text_embedding,
            "audio_embedding": audio_embedding,
            "video_embedding": video_embedding
        }

    # Save the embeddings to a pickle file
    with open(os.path.join(save_path, 'multi_model_embeddings.pkl'), 'wb') as f:
        pickle.dump(multi_model_embeddings, f)
    print("Embeddings have been saved successfully.")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py <audio_dir> <text_emb_dir> <audio_emb_dir> <video_emb_dir> <save_path>")
    else:
        main(*sys.argv[1:])
