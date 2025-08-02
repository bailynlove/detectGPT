import joblib, torch
from utils import detectgpt_score, style_feats, deberta_tok, deberta_model

# 加载微调后的 DeBERTa
deberta_model.load_state_dict(torch.load("models/deberta_ft/pytorch_model.bin", map_location="cuda"))
fusion = joblib.load("models/fusion.pkl")

def ai_score(text: str) -> float:
    f_dg = detectgpt_score(text, k=2)
    sty = style_feats(text)
    inp = deberta_tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.no_grad():
        prob_ai = float(torch.softmax(deberta_model(**inp).logits, dim=-1)[0, 1])
    feat = [f_dg, prob_ai, sty["avg_sent_len"], sty["ttr"], sty["punct_ratio"]]
    return float(fusion.predict([feat])[0])

if __name__ == "__main__":
    while True:
        txt = input("输入文本：").strip()
        if not txt:
            break
        print("AI 率：", ai_score(txt))