from fastapi import FastAPI, UploadFile, File
import numpy as np
import pandas as pd
import torch
import imagehash

from utils import get_clip_model, extract_embedding, extract_hash
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# 데이터 로딩
labels_df = pd.read_csv('labels_with_hash.csv')
embeddings = np.load('embeddings.npy')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = get_clip_model(device)

THRESHOLD = 0.75  # 임계값(실험 후 보정)
HASH_MATCH = True  # hash로 완전 매칭 우선

@app.post("/match")
async def match_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    from io import BytesIO
    from PIL import Image

    img = Image.open(BytesIO(img_bytes)).convert('RGB')

    # Perceptual hash 완전 매칭
    if HASH_MATCH:
        query_hash = str(imagehash.phash(img))
        matched = labels_df[labels_df['hash'] == query_hash]
        if not matched.empty:
            return {"result": matched.iloc[0]['label'], "type": "hash_match"}

    # CLIP 임베딩 유사도
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor).cpu().numpy()[0]
    sims = cosine_similarity([emb], embeddings)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]
    if best_score >= THRESHOLD:
        return str(labels_df.iloc[best_idx]['label'])
    else:
        return "미등록"
