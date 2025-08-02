import pandas as pd, joblib, torch
from utils import detectgpt_score, style_feats, deberta_tok, deberta_model
from tqdm import tqdm

df = pd.read_csv("data/full_data_0731_aug_4.csv")[["text", "prompt_type"]].dropna()
df["label"] = (df["prompt_type"] != 0).astype(int)

features, labels = [], []
for text, label in tqdm(df.itertuples(index=False), total=len(df)):
    f_dg = detectgpt_score(text, k=2)                # DetectGPT
    sty = style_feats(text)                          # 风格
    # DeBERTa 概率
    inp = deberta_tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        prob_ai = float(torch.softmax(deberta_model(**inp).logits, dim=-1)[0, 1])
    feat = [f_dg, prob_ai, sty["avg_sent_len"], sty["ttr"], sty["punct_ratio"]]
    features.append(feat)
    labels.append(label)

joblib.dump((features, labels), "cache/features.pkl")