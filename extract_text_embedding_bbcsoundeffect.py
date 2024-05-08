import os
import json
import torch
import numpy as np
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

def extract_bbc_sound_effect_text_embeddings(audio_dir, json_path, output_dir):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    clip_type = {'video': 'LanguageBind_Video_V1.5_FT'}
    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    batch_size = 512
    for i in range(0, len(data['data']), batch_size):
        batch_data = data['data'][i:i+batch_size]
        
        captions = [item['caption'] for item in batch_data if f"{item['id']}.flac" in os.listdir(audio_dir)]
        if not captions:
            continue
        
        inputs = to_device(tokenizer(captions, max_length=77, padding='max_length', 
                                     truncation=True, return_tensors='pt'), device)
        
        with torch.no_grad():
            embeddings = model({'language': inputs})['language']
        
        for j, item in enumerate(batch_data):
            if f"{item['id']}.flac" in os.listdir(audio_dir):
                np.save(os.path.join(output_dir, f"{item['id']}.npy"), embeddings[j].cpu().numpy())

if __name__ == '__main__':
    audio_dir = '/workspace/mnt/mm_data/libingliang/BBC_new/audio_441_stereo'
    json_path = '/workspace/mnt/mm_data/libingliang/BBC_new/bbc_final.json'
    output_dir = '/workspace/mnt/mm_data/libingliang/BBC_new/text_embedding'
    
    extract_bbc_sound_effect_text_embeddings(audio_dir, json_path, output_dir)
