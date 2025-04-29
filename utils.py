import open_clip
import torch
from PIL import Image
import numpy as np
import imagehash

def get_clip_model(device='cpu'):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        device=device,
        cache_dir="./model_cache"  # 👈 여기가 핵심!
    )
    model.eval()
    return model, preprocess

def extract_embedding(image_path, model, preprocess, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
    return emb.cpu().numpy()[0]

def extract_hash(image_path):
    img = Image.open(image_path).convert('RGB')
    return str(imagehash.phash(img))  # perceptual hash
