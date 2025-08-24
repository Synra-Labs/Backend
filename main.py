#!/usr/bin/env python3
import os, json, uuid, time, logging, io
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Optional deps – only used when you actually load locally / train
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

logging.basicConfig(level=logging.INFO)

# ─────────────────────────── Env
ALLOW_ORIGINS  = os.getenv("ALLOW_ORIGINS", "*")
FORCE_REMOTE   = os.getenv("FORCE_REMOTE", "0") == "1"   # Force HF/OpenAI only, no local loading
HF_API_TOKEN   = os.getenv("HF_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_ENDPOINT    = os.getenv("S3_ENDPOINT", "")
AWS_REGION     = os.getenv("AWS_REGION", "auto")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/tmp/nspire_models"; os.makedirs(CACHE_DIR, exist_ok=True)

# Optional S3 client (only used for training artifacts / downloads)
s3 = None
if S3_BUCKET:
    s3 = boto3.client(
        "s3",
        endpoint_url=(S3_ENDPOINT or None),
        aws_access_key_id=(AWS_ACCESS_KEY_ID or None),
        aws_secret_access_key=(AWS_SECRET_ACCESS_KEY or None),
        region_name=(AWS_REGION if AWS_REGION != "auto" else None),
    )

# ─────────────────────────── Data stores (simple in-memory; swap to Redis later if you want)
# Per-session configs and trained models. Keyed by session id, then model id.
CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {}
LOCAL_MODELS: Dict[str, Any] = {}  # loaded pipelines (base + finetunes)
TRAINED: Dict[str, Dict[str, Any]] = {}  # {session_id: {model_id: meta}}  meta["s3_uri"]

# ─────────────────────────── Curated base models (label → HF repo id)
BASE_MODELS = [
    {"label":"TinyLlama-1.1B-Chat", "repo":"TinyLlama/TinyLlama-1.1B-Chat-v1.0", "public": True},
    {"label":"Mistral-7B-Instruct-v0.3", "repo":"mistralai/Mistral-7B-Instruct-v0.3", "public": True},
    {"label":"CodeLlama-7B-Instruct", "repo":"codellama/CodeLlama-7b-Instruct-hf", "public": True},
    {"label":"DeepSeek-LLM-7B-Chat", "repo":"deepseek-ai/DeepSeek-LLM-7B-Chat", "public": True},
    {"label":"Phi-3-mini-4k-instruct", "repo":"microsoft/Phi-3-mini-4k-instruct", "public": True},
    {"label":"Qwen2-1.5B-Instruct", "repo":"Qwen/Qwen2-1.5B-Instruct", "public": True},
    {"label":"Gemma-2-2B-it", "repo":"google/gemma-2-2b-it", "public": True},
    # Gated/Llama3 variants will work only if your HF token has access:
    {"label":"LLaMA-3-8B-Instruct (gated)", "repo":"meta-llama/Meta-Llama-3-8B-Instruct", "public": False},
]

# ─────────────────────────── FastAPI
app = FastAPI(title="Nspire API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == "*" else [ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────── Session helper (no login; per-browser header)
def session(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or f"anon-{uuid.uuid4()}"

# ─────────────────────────── Schemas
class ModifyChat(BaseModel):
    model_id: str = Field(..., alias="modelId")
    temperature: float = 0.7
    token_limit: int = Field(256, alias="tokenLimit")
    instructions: str = Field("", alias="instructions")
    top_p: Optional[float] = Field(None, alias="topP")
    top_k: Optional[int]   = Field(None, alias="topK")
    stop: Optional[List[str]] = None
    model_config = {"populate_by_name": True, "protected_namespaces": ()}

class RunChat(BaseModel):
    model_id: str = Field(..., alias="modelId")
    prompt: str
    model_config = {"populate_by_name": True, "protected_namespaces": ()}

# ─────────────────────────── Small helpers
def _defaults() -> Dict[str, Any]:
    return {"temperature":0.7,"max_tokens":256,"instructions":"","top_p":"","top_k":"","stop":""}

def _get_cfg(sid: str, mid: str) -> Dict[str, Any]:
    return CONFIGS.get(sid, {}).get(mid, _defaults())

def _set_cfg(sid: str, mid: str, cfg: Dict[str, Any]):
    CONFIGS.setdefault(sid, {})[mid] = cfg

def _build_prompt(instr: str, user: str) -> str:
    i = (instr or "").strip()
    return f"{i}\n{user}" if i else user

def _gen_kwargs(cfg: Dict[str,Any]) -> Dict[str,Any]:
    out = {
        "max_new_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "do_sample": True,
        "return_full_text": False
    }
    if cfg.get("top_p") not in ("", None): out["top_p"] = float(cfg["top_p"])
    if cfg.get("top_k") not in ("", None): out["top_k"] = int(cfg["top_k"])
    return out

def _ensure_loaded_base(hf_id: str):
    if hf_id in LOCAL_MODELS: return
    if FORCE_REMOTE:
        raise RuntimeError("FORCE_REMOTE=1: local load disabled")
    logging.info(f"Loading base HF model locally: {hf_id}")
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        device_map="auto" if DEVICE=="cuda" else None
    )
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=(0 if DEVICE=="cuda" else -1))
    LOCAL_MODELS[hf_id] = {"pipe": pipe, "meta":{"type":"base"}}

def _download_s3_prefix(local_dir: str, bucket: str, prefix: str):
    os.makedirs(local_dir, exist_ok=True)
    token = None
    while True:
        kw={"Bucket":bucket,"Prefix":prefix}
        if token: kw["ContinuationToken"]=token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            key=obj["Key"]; rel=key[len(prefix):].lstrip("/")
            dest=os.path.join(local_dir, rel); os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
        if resp.get("IsTruncated"):
            token=resp["NextContinuationToken"]
        else:
            break

def _ensure_loaded_finetune(model_id: str, meta: Dict[str, Any]):
    if model_id in LOCAL_MODELS: return
    if FORCE_REMOTE:
        raise RuntimeError("FORCE_REMOTE=1: local load disabled")
    s3_uri = meta.get("s3_uri")
    base   = meta.get("base_model_id")
    if not (s3 and s3_uri and base):
        raise RuntimeError("Fine-tune meta incomplete (need s3_uri + base_model_id)")

    # pull adapter from S3
    _,_,rest = s3_uri.partition("s3://"); bucket,_,prefix = rest.partition("/")
    local_dir = os.path.join(CACHE_DIR, f"ft-{model_id}")
    _download_s3_prefix(local_dir, bucket or S3_BUCKET, prefix)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto" if DEVICE=="cuda" else None
    )
    model = PeftModel.from_pretrained(base_model, local_dir)
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=(0 if DEVICE=="cuda" else -1))
    LOCAL_MODELS[model_id] = {"pipe": pipe, "meta":{"type":"lora","base":base}}

# ─────────────────────────── Routes
@app.get("/")
def health():
    return {"ok": True, "device": DEVICE, "message": "Nspire API live"}

@app.get("/debug/backends")
def debug_backends(sid: str = Depends(session)):
    return {
        "device": DEVICE,
        "have_hf_token": bool(HF_API_TOKEN),
        "have_openai_key": bool(OPENAI_API_KEY),
        "force_remote": FORCE_REMOTE,
        "local_models_loaded": list(LOCAL_MODELS.keys()),
        "session_trained_models": list(TRAINED.get(sid, {}).keys()),
    }

@app.get("/models")
def list_models():
    """Return curated base models (labels + HF repo IDs)."""
    return {"models": BASE_MODELS}

# GET current config (optionally download), POST to save config (modify)
@app.get("/config")
def get_config(model_id: str = Query(...), download: bool = Query(False), sid: str = Depends(session)):
    cfg = _get_cfg(sid, model_id)
    if download:
        buf = io.BytesIO(json.dumps({"modelId":model_id, **cfg}, indent=2).encode("utf-8"))
        headers = {"Content-Disposition": f'attachment; filename="{model_id.replace("/","_")}-config.json"'}
        return StreamingResponse(buf, media_type="application/json", headers=headers)
    return {"modelId": model_id, "config": cfg}

@app.post("/config")
@app.post("/modify-file")  # alias for your existing frontend
def post_config(
    req: ModifyChat,
    prewarm: bool = Query(False),
    sid: str = Depends(session)
):
    cfg = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
        "top_p": req.top_p if req.top_p is not None else "",
        "top_k": req.top_k if req.top_k is not None else "",
        "stop": json.dumps(req.stop) if req.stop else "",
    }
    _set_cfg(sid, req.model_id, cfg)

    # Optional local pre-load
    if prewarm and not FORCE_REMOTE:
        try:
            _ensure_loaded_base(req.model_id)
        except Exception as e:
            logging.warning(f"Prewarm failed (continuing anyway): {e}")

    return {"success": True, "message": "Config saved", "modelId": req.model_id}

@app.post("/run")
def run(req: RunChat, sid: str = Depends(session)):
    mid = req.model_id
    cfg = _get_cfg(sid, mid)
    prompt = _build_prompt(cfg.get("instructions",""), req.prompt)

    # helpers
    def hf_remote():
        if not HF_API_TOKEN: return None
        try:
            url = f"https://api-inference.huggingface.co/models/{mid}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": float(cfg.get("temperature",0.7)),
                    "max_new_tokens": int(cfg.get("max_tokens",256))
                }
            }
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"]
            if isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            return str(data)
        except Exception as e:
            logging.warning(f"HF inference failed: {e}")
            return None

    def openai_fallback():
        if not OPENAI_API_KEY: return None
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            messages=[]
            if cfg.get("instructions"): messages.append({"role":"system","content":cfg["instructions"]})
            messages.append({"role":"user","content":req.prompt})
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=float(cfg.get("temperature",0.7)),
                max_tokens=int(cfg.get("max_tokens",256))
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI fallback failed: {e}")
            return None

    # Fast path for tiny hosts
    if FORCE_REMOTE:
        text = hf_remote() or openai_fallback()
        if text is not None:
            src = "hf_inference" if HF_API_TOKEN and text else "openai_fallback"
            return {"success": True, "source": src, "response": text}
        raise HTTPException(502, detail="No remote backend available. Set HF_API_TOKEN or OPENAI_API_KEY.")

    # 1) fine-tune loaded for this session?
    meta = TRAINED.get(sid, {}).get(mid)
    if meta:
        try:
            _ensure_loaded_finetune(mid, meta)
            pipe = LOCAL_MODELS[mid]["pipe"]
            out = pipe(prompt, **_gen_kwargs(cfg))
            return {"success": True, "source": "finetune-local", "response": out[0]["generated_text"].strip()}
        except Exception as e:
            logging.error(f"Local finetune run failed: {e}")

    # 2) prewarmed base?
    if mid in LOCAL_MODELS:
        try:
            pipe = LOCAL_MODELS[mid]["pipe"]
            out = pipe(prompt, **_gen_kwargs(cfg))
            return {"success": True, "source": "hf-local", "response": out[0]["generated_text"].strip()}
        except Exception as e:
            logging.error(f"Local prewarmed run failed: {e}")

    # 3) HF hosted inference
    text = hf_remote()
    if text is not None:
        return {"success": True, "source": "hf_inference", "response": text}

    # 4) Try local on-demand (can OOM!)
    try:
        _ensure_loaded_base(mid)
        pipe = LOCAL_MODELS[mid]["pipe"]
        out = pipe(prompt, **_gen_kwargs(cfg))
        return {"success": True, "source": "hf-local-on-demand", "response": out[0]["generated_text"].strip()}
    except Exception as e:
        logging.error(f"Local on-demand load failed: {e}")

    # 5) OpenAI
    text = openai_fallback()
    if text is not None:
        return {"success": True, "source": "openai_fallback", "response": text}

    raise HTTPException(502, detail="No inference backend available. Provide HF_API_TOKEN or OPENAI_API_KEY.")

# ─────────────────────────── Training (simplified; replace with your queue/RunPod if needed)
@app.post("/train")
async def train(
    base_model_id: str = Form(...),
    files: List[UploadFile] = File(...),
    sid: str = Depends(session)
):
    # Here you’d kick a real trainer. For demo, register a fake finetune pointing to S3 prefix you later upload.
    texts=[]
    for f in files:
        b=await f.read()
        s=b.decode("utf-8", errors="ignore").strip()
        if s: texts.append(s)
    if not texts:
        raise HTTPException(400, "No valid text provided")

    job_id = str(uuid.uuid4())
    # In a real job you’d stream progress; we simulate "in_progress"
    TRAINED.setdefault(sid, {})
    TRAINED[sid][f"ft-{job_id}"] = {
        "s3_uri": "",        # fill when your trainer uploads to s3://bucket/prefix
        "base_model_id": base_model_id,
        "type": "lora",
        "created": int(time.time()),
    }
    return {"job_id": job_id, "status": "in_progress"}

@app.get("/progress/{job_id}")
def progress(job_id: str, sid: str = Depends(session)):
    # Replace with your real job polling. Here we just flip to "completed" quickly.
    ft_id = f"ft-{job_id}"
    meta = TRAINED.get(sid, {}).get(ft_id)
    if not meta:
        raise HTTPException(404, "Job not found")
    # pretend finished:
    meta.setdefault("s3_uri", f"s3://{S3_BUCKET}/models/{sid}/{ft_id}/")  # your trainer should actually write here
    return {"status": "completed", "model_id": ft_id, "meta": meta}

@app.get("/download-model")
def download_model(model_id: str = Query(...), sid: str = Depends(session)):
    """Return presigned URLs for all files of a trained model (S3)."""
    meta = TRAINED.get(sid, {}).get(model_id)
    if not (s3 and meta and meta.get("s3_uri")):
        raise HTTPException(404, "No S3 artifact for this model")
    _,_,rest = meta["s3_uri"].partition("s3://"); bucket,_,prefix = rest.partition("/")
    if not bucket: bucket = S3_BUCKET
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    items = []
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600
        )
        items.append({"key": key, "url": url, "size": obj.get("Size", 0)})
    return {"model_id": model_id, "files": items}

# ─────────────────────────── Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
