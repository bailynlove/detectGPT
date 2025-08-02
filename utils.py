import torch, re, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import List

torch.cuda.empty_cache()

# ---------- Qwen3-8B ----------
qwen_name = "Qwen/Qwen3-8B-Instruct"
qwen_tok = AutoTokenizer.from_pretrained(qwen_name, cache_dir="cache")
qwen_tok.pad_token = qwen_tok.eos_token
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_name,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="cache",
    trust_remote_code=True,
)


@torch.no_grad()
def get_logprob(text: str) -> float:
    inputs = qwen_tok(text, return_tensors="pt").to(qwen_model.device)
    labels = inputs.input_ids.clone()
    loss = qwen_model(**inputs, labels=labels).loss
    return -loss.item()


# ---------- T5-small 扰动 ----------
from transformers import T5Tokenizer, T5ForConditionalGeneration

t5_tok = T5Tokenizer.from_pretrained("t5-small", cache_dir="cache")
t5_model = T5ForConditionalGeneration.from_pretrained(
    "t5-small", torch_dtype=torch.float16, device_map="auto", cache_dir="cache"
)


@torch.no_grad()
def perturb(text: str, k: int = 3) -> List[str]:
    tokens = text.split()
    n = max(1, int(len(tokens) * 0.15))
    outs = []
    for _ in range(k):
        idxs = np.random.choice(len(tokens), n, replace=False)
        masked = [tok if i not in idxs else "<extra_id_0>" for i, tok in enumerate(tokens)]
        inp = t5_tok(" ".join(masked), return_tensors="pt").input_ids.to(t5_model.device)
        gen = t5_model.generate(inp, max_length=len(tokens) + 10, do_sample=True, top_p=0.9, temperature=0.8)
        outs.append(t5_tok.decode(gen[0], skip_special_tokens=True))
    torch.cuda.empty_cache()
    return outs


def detectgpt_score(text: str, k: int = 3) -> float:
    orig = get_logprob(text)
    perturbed = perturb(text, k)
    pert_scores = [get_logprob(p) for p in perturbed]
    return orig - np.mean(pert_scores)


# ---------- DeBERTa ----------
deberta_name = "microsoft/deberta-v3-base"
deberta_tok = AutoTokenizer.from_pretrained(deberta_name, cache_dir="cache")
deberta_model = AutoModelForSequenceClassification.from_pretrained(
    deberta_name, num_labels=2, torch_dtype=torch.float16, cache_dir="cache"
).to("cuda")


# ---------- 风格特征 ----------
def style_feats(text: str):
    tokens = text.split()
    sents = re.split(r"[.!?]", text)
    return {
        "avg_sent_len": np.mean([len(s.split()) for s in sents if s.strip()]),
        "ttr": len(set(tokens)) / max(1, len(tokens)),
        "punct_ratio": sum(1 for c in text if not c.isalnum()) / max(1, len(text)),
    }
