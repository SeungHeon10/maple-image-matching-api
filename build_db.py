import pandas as pd
import os
import numpy as np
from utils import get_clip_model, extract_embedding, extract_hash
import torch

def build_db(image_folder='C:/Users/hun00/Downloads/dataset', labels_csv='labels.csv'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = get_clip_model(device)
    df = pd.read_csv(labels_csv)
    embs = []
    hashes = []
    for idx, fname in enumerate(df['filename']):
        path = os.path.join(image_folder, fname)
        emb = extract_embedding(path, model, preprocess, device)
        embs.append(emb)
        hashes.append(extract_hash(path))
        print(f"[{idx+1}/{len(df)}] {fname} done")  # <-- 진행상황 표시
    np.save('embeddings.npy', np.array(embs))
    df['hash'] = hashes
    df.to_csv('labels_with_hash.csv', index=False)

if __name__ == "__main__":
    build_db()
