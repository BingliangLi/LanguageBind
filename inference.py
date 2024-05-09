import torch
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer

if __name__ == '__main__':
    device = 'cuda:0'
    device = torch.device(device)
    clip_type = {
        'video': 'LanguageBind_Video_V1.5_FT',  # also LanguageBind_Video
        'audio': 'LanguageBind_Audio_FT',  # also LanguageBind_Audio
    }

    model = LanguageBind(clip_type=clip_type, cache_dir='./cache_dir')
    model = model.to(device)
    model.eval()
    pretrained_ckpt = f'LanguageBind/LanguageBind_Image'
    tokenizer = LanguageBindImageTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir/tokenizer_cache_dir')
    modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in clip_type.keys()}

    audio = ['assets/audio/0.wav']
    video = ['/workspace/mnt/mm_data/libingliang/VGGSound/video/ljjUj5fQZgs_000450.mp4']
    language = ["Training a parakeet to climb up a ladder."]

    inputs = {
        'video': to_device(modality_transform['video'](video), device),
        'audio': to_device(modality_transform['audio'](audio), device),
    }
    inputs['language'] = to_device(tokenizer(language, max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)

    with torch.no_grad():
        embeddings = model(inputs)

    print("Video x Text: \n",
          torch.softmax(embeddings['video'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    print("Audio x Text: \n",
          torch.softmax(embeddings['audio'] @ embeddings['language'].T, dim=-1).detach().cpu().numpy())
    print("Video x Audio: \n",
          torch.softmax(embeddings['video'] @ embeddings['audio'].T, dim=-1).detach().cpu().numpy())
    
    # save embeddings['video'] to /workspace/mnt/mm_data/libingliang/VGGSound/video_embedding/ljjUj5fQZgs_000450_video_embedding.npy
    emb = embeddings['video'].cpu().numpy()
    output_path = '/workspace/mnt/mm_data/libingliang/VGGSound/video_embedding/ljjUj5fQZgs_000450_video_embedding.npy'
    np.save(output_path, emb)
    

