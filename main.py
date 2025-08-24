#!/usr/bin/env python3
"""
Nspire API (FastAPI) — chat, config(modify), optional training stub, downloads.

Env you can set on Railway:
  - OPENAI_API_KEY           # enable OpenAI fallback
  - HF_API_TOKEN             # enable Hugging Face Hosted Inference
  - FORCE_REMOTE=1           # recommended on small hosts: skip local model loads
  - ALLOW_ORIGINS=*          # CORS (set to your Vercel domain in prod)

Optional (for S3/R2 downloads after training):
  - S3_BUCKET, S3_ENDPOINT, AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""

import os, io, json, uuid, time, logging
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Optional local loading (disabled when FORCE_REMOTE=1)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
except Exception:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = pipeline = PeftModel = None  # type: ignore

# Optional S3/R2
try:
    import boto3
except Exception:
    boto3 = None  # type: ignore

logging.basicConfig(level=logging.INFO)

# ─────────── Env
ALLOW_ORIGINS  = os.getenv("ALLOW_ORIGINS", "*")
FORCE_REMOTE   = os.getenv("FORCE_REMOTE", "0") == "1"
HF_API_TOKEN   = os.getenv("HF_API_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

S3_BUCKET      = os.getenv("S3_BUCKET", "")
S3_ENDPOINT    = os.getenv("S3_ENDPOINT", "")
AWS_REGION     = os.getenv("AWS_REGION", "auto")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

DEVICE = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
CACHE_DIR = "/tmp/nspire_models"; os.makedirs(CACHE_DIR, exist_ok=True)

# Optional S3 client
s3 = None
if S3_BUCKET and boto3:
    s3 = boto3.client(
        "s3",
        endpoint_url=(S3_ENDPOINT or None),
        aws_access_key_id=(AWS_ACCESS_KEY_ID or None),
        aws_secret_access_key=(AWS_SECRET_ACCESS_KEY or None),
        region_name=(None if AWS_REGION == "auto" else AWS_REGION),
    )

# ─────────── In-memory stores (per-container)
CONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {}      # session_id -> model_id -> cfg
LOCAL_MODELS: Dict[str, Any] = {}                        # model_id -> {"pipe":..., "meta":...}
TRAINED: Dict[str, Dict[str, Any]] = {}                  # session_id -> {ft_id: meta}
JOBS: Dict[str, Dict[str, Any]] = {}                     # session_id -> {job_id: {...}}

# ─────────── Curated base models (label + HF repo id)
BASE_MODELS = [
    {"label":"TinyLlama-1.1B-Chat",         "repo":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",   "public": True},
    {"label":"Phi-3-mini-4k-instruct",      "repo":"microsoft/Phi-3-mini-4k-instruct",     "public": True},
    {"label":"Qwen2-1.5B-Instruct",         "repo":"Qwen/Qwen2-1.5B-Instruct",             "public": True},
    {"label":"Gemma-2-2B-it",               "repo":"google/gemma-2-2b-it",                 "public": True},
    {"label":"Mistral-7B-Instruct-v0.3",    "repo":"mistralai/Mistral-7B-Instruct-v0.3",   "public": True},
    {"label":"CodeLlama-7B-Instruct",       "repo":"codellama/CodeLlama-7b-Instruct-hf",   "public": True},
    {"label":"DeepSeek-LLM-7B-Chat",        "repo":"deepseek-ai/deepseek-llm-7b-chat",     "public": True},
    {"label":"DeepSeek-Coder-1.3B-Instruct","repo":"deepseek-ai/deepseek-coder-1.3b-instruct","public": True},
    {"label":"Falcon-7B (base)",            "repo":"tiiuae/falcon-7b",                      "public": True},
    {"label":"GPT-J 6B (base)",             "repo":"EleutherAI/gpt-j-6B",                   "public": True},
    {"label":"DistilGPT-2 (tiny)",          "repo":"distilgpt2",                            "public": True},
    {"label":"LLaMA-3-8B-Instruct (gated)", "repo":"meta-llama/Meta-Llama-3-8B-Instruct",  "public": False},
]

# ─────────── FastAPI
app = FastAPI(title="Nspire API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == "*" else [ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────── Session (no login; per-browser header)
def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or f"anon-{uuid.uuid4()}"

# ─────────── Schemas (pydantic v2 compatible)
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

# ─────────── Helpers
def _defaults() -> Dict[str, Any]:
    return {"temperature":0.7,"max_tokens":256,"instructions":"","top_p":"","top_k":"","stop":""}

def _cfg(sid: str, mid: str) -> Dict[str, Any]:
    return CONFIGS.get(sid, {}).get(mid, _defaults())

def _set_cfg(sid: str, mid: str, cfg: Dict[str, Any]):
    CONFIGS.setdefault(sid, {})[mid] = cfg

def _build_prompt(sys: str, user: str) -> str:
    sys = (sys or "").strip()
    return f"{sys}\n{user}" if sys else user

def _gen_kwargs(cfg: Dict[str,Any]) -> Dict[str,Any]:
    out = {
        "max_new_tokens": int(cfg.get("max_tokens",256)),
        "temperature": float(cfg.get("temperature",0.7)),
        "do_sample": True,
        "return_full_text": False
    }
    if cfg.get("top_p") not in ("", None): out["top_p"] = float(cfg["top_p"])
    if cfg.get("top_k") not in ("", None): out["top_k"] = int(cfg["top_k"])
    return out

# HF hosted inference with robust behavior (wait_for_model & clean 404 handling)
def hf_remote_infer(mid: str, prompt: str, cfg: Dict[str,Any]) -> Optional[str]:
    if not HF_API_TOKEN:
        return None
    try:
        url = f"https://api-inference.huggingface.co/models/{mid}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": float(cfg.get("temperature",0.7)),
                "max_new_tokens": int(cfg.get("max_tokens",256)),
            },
            "options": {"wait_for_model": True, "use_cache": True},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code == 404:
            logging.warning(f"HF 404 for repo {mid} — check repo id / token access.")
            return None
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)
    except Exception as e:
        logging.warning(f"HF inference failed for {mid}: {e}")
        return None

# OpenAI fallback that works with both v1 and legacy v0 SDKs
def openai_fallback_chat(prompt_only: str, sys: str, cfg: Dict[str,Any]) -> Optional[str]:
    if not OPENAI_API_KEY:
        return None
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # avoid passing kwargs like proxies
        # Try modern SDK (openai>=1.x)
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI()
            messages = []
            if sys: messages.append({"role":"system","content":sys})
            messages.append({"role":"user","content":prompt_only})
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=float(cfg.get("temperature",0.7)),
                max_tokens=int(cfg.get("max_tokens",256)),
            )
            return resp.choices[0].message.content.strip()
        except Exception as e_v1:
            logging.warning(f"OpenAI v1 path failed ({e_v1}), trying legacy v0...")
            # Legacy SDK (openai<1.0)
            import openai as openai_v0  # type: ignore
            openai_v0.api_key = OPENAI_API_KEY
            messages = []
            if sys: messages.append({"role":"system","content":sys})
            messages.append({"role":"user","content":prompt_only})
            resp = openai_v0.ChatCompletion.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                temperature=float(cfg.get("temperature",0.7)),
                max_tokens=int(cfg.get("max_tokens",256)),
            )
            return resp.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"OpenAI fallback failed: {e}")
        return None

# Local loading utilities (used only when FORCE_REMOTE=0)
def _ensure_loaded_base(hf_id: str):
    if FORCE_REMOTE:
        raise RuntimeError("FORCE_REMOTE=1: local load disabled")
    if hf_id in LOCAL_MODELS:
        return
    if not (AutoTokenizer and AutoModelForCausalLM and pipeline):
        raise RuntimeError("Transformers not installed for local loading.")
    logging.info(f"Loading base HF model locally: {hf_id}")
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, device_map="auto" if DEVICE=="cuda" else None
    )
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=(0 if DEVICE=="cuda" else -1))
    LOCAL_MODELS[hf_id] = {"pipe": pipe, "meta":{"type":"base"}}

def _download_s3_prefix(local_dir: str, bucket: str, prefix: str):
    if not s3:
        raise RuntimeError("S3 client not configured")
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
    if FORCE_REMOTE:
        raise RuntimeError("FORCE_REMOTE=1: local load disabled")
    if model_id in LOCAL_MODELS:
        return
    if not (AutoTokenizer and AutoModelForCausalLM and PeftModel and pipeline):
        raise RuntimeError("Transformers/peft not installed for local loading.")
    s3_uri = meta.get("s3_uri"); base = meta.get("base_model_id")
    if not (s3 and s3_uri and base):
        raise RuntimeError("Fine-tune meta incomplete (need s3_uri + base_model_id)")
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

# ─────────── Routes
@app.get("/")
def health():
    return {"ok": True, "device": DEVICE, "force_remote": FORCE_REMOTE, "message": "Nspire API live"}

@app.get("/debug/backends")
def debug_backends(sid: str = Depends(get_session_id)):
    return {
        "device": DEVICE,
        "have_hf_token": bool(HF_API_TOKEN),
        "have_openai_key": bool(OPENAI_API_KEY),
        "force_remote": FORCE_REMOTE,
        "local_models_loaded": list(LOCAL_MODELS.keys()),
        "trained_models_for_session": list(TRAINED.get(sid, {}).keys()),
    }

@app.get("/models")
def list_models():
    return {"models": BASE_MODELS}

@app.get("/config")
def get_config(model_id: str = Query(...), download: bool = Query(False), sid: str = Depends(get_session_id)):
    cfg = _cfg(sid, model_id)
    if download:
        buf = io.BytesIO(json.dumps({"modelId":model_id, **cfg}, indent=2).encode("utf-8"))
        headers = {"Content-Disposition": f'attachment; filename="{model_id.replace("/","_")}-config.json"'}
        return StreamingResponse(buf, media_type="application/json", headers=headers)
    return {"modelId": model_id, "config": cfg}

@app.post("/config")
@app.post("/modify-file")  # alias for your existing frontend
def post_config(req: ModifyChat, prewarm: bool = Query(False), sid: str = Depends(get_session_id)):
    cfg = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
        "top_p": req.top_p if req.top_p is not None else "",
        "top_k": req.top_k if req.top_k is not None else "",
        "stop": json.dumps(req.stop) if req.stop else "",
    }
    _set_cfg(sid, req.model_id, cfg)

    if prewarm and not FORCE_REMOTE:
        try:
            _ensure_loaded_base(req.model_id)
        except Exception as e:
            logging.warning(f"Prewarm failed (continuing anyway): {e}")

    return {"success": True, "message": "Config saved", "modelId": req.model_id}

@app.post("/run")
def run(req: RunChat, sid: str = Depends(get_session_id)):
    mid = req.model_id
    cfg = _cfg(sid, mid)
    sys = (cfg.get("instructions") or "").strip()
    prompt_only = req.prompt
    full_prompt = _build_prompt(sys, prompt_only)

    # Remote-only on small hosts
    if FORCE_REMOTE:
        text = hf_remote_infer(mid, full_prompt, cfg)
        if text is not None:
            return {"success": True, "source": "hf_inference", "response": text}
        text = openai_fallback_chat(prompt_only, sys, cfg)
        if text is not None:
            return {"success": True, "source": "openai_fallback", "response": text}
        raise HTTPException(502, detail="No remote backend available. Set HF_API_TOKEN or OPENAI_API_KEY.")

    # Try local fine-tune for this session
    meta = TRAINED.get(sid, {}).get(mid)
    if meta:
        try:
            _ensure_loaded_finetune(mid, meta)
            pipe = LOCAL_MODELS[mid]["pipe"]
            out = pipe(full_prompt, **_gen_kwargs(cfg))
            return {"success": True, "source": "finetune-local", "response": out[0]["generated_text"].strip()}
        except Exception as e:
            logging.error(f"Local finetune run failed: {e}")

    # Prewarmed base?
    if mid in LOCAL_MODELS:
        try:
            pipe = LOCAL_MODELS[mid]["pipe"]
            out = pipe(full_prompt, **_gen_kwargs(cfg))
            return {"success": True, "source": "hf-local", "response": out[0]["generated_text"].strip()}
        except Exception as e:
            logging.error(f"Local prewarmed run failed: {e}")

    # Hosted HF
    text = hf_remote_infer(mid, full_prompt, cfg)
    if text is not None:
        return {"success": True, "source": "hf_inference", "response": text}

    # Try local on-demand (may OOM)
    try:
        _ensure_loaded_base(mid)
        pipe = LOCAL_MODELS[mid]["pipe"]
        out = pipe(full_prompt, **_gen_kwargs(cfg))
        return {"success": True, "source": "hf-local-on-demand", "response": out[0]["generated_text"].strip()}
    except Exception as e:
        logging.error(f"Local on-demand load failed: {e}")

    # OpenAI fallback
    text = openai_fallback_chat(prompt_only, sys, cfg)
    if text is not None:
        return {"success": True, "source": "openai_fallback", "response": text}

    raise HTTPException(502, detail="No inference backend available. Provide HF_API_TOKEN or OPENAI_API_KEY.")

# ─────────── Minimal training stub (registers a fake finetune so the UI flows)
@app.post("/train")
async def train(
    base_model_id: str = Form(...),
    files: List[UploadFile] = File(...),
    sid: str = Depends(get_session_id),
):
    texts=[]
    for f in files:
        b=await f.read()
        s=b.decode("utf-8", errors="ignore").strip()
        if s: texts.append(s)
    if not texts:
        raise HTTPException(400, "No valid text provided")

    job_id = str(uuid.uuid4())
    JOBS.setdefault(sid, {})[job_id] = {"status":"in_progress", "ts": time.time()}
    # Create an ft id & placeholder meta (your real trainer should overwrite s3_uri later)
    ft_id = f"ft-{job_id}"
    TRAINED.setdefault(sid, {})[ft_id] = {
        "s3_uri": f"s3://{S3_BUCKET}/models/{sid}/{ft_id}/",  # placeholder prefix
        "base_model_id": base_model_id,
        "type": "lora",
        "created": int(time.time()),
    }
    return {"job_id": job_id, "status":"in_progress"}

@app.get("/progress/{job_id}")
def progress(job_id: str, sid: str = Depends(get_session_id)):
    job = JOBS.get(sid, {}).get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    # Simulate quick completion for demo
    if time.time() - job["ts"] > 2:
        job["status"] = "completed"
    ft_id = f"ft-{job_id}"
    meta = TRAINED.get(sid, {}).get(ft_id, {})
    return {"status": job["status"], "model_id": ft_id, "meta": meta}

@app.get("/download-model")
def download_model(model_id: str = Query(...), sid: str = Depends(get_session_id)):
    """Return presigned URLs for trained model files if S3 is configured."""
    if not s3:
        raise HTTPException(400, "S3 not configured")
    meta = TRAINED.get(sid, {}).get(model_id)
    if not (meta and meta.get("s3_uri")):
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

# ─────────── Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
