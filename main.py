#!/usr/bin/env python3
"""
NSPIRE API
- /models           → list base HF models + your session's trained models
- /modify-file      → save real per-model generation config (applied at /run)
- /train            → queue LoRA/full training on RunPod (uploads are text)
- /progress/{job}   → poll RunPod; when done, register model (s3://...) for session
- /load-local       → (optional) pre-load a trained model into RAM
- /run              → chat using: local trained → HF hosted → OpenAI fallback
"""

import os, json, uuid, time, logging
from typing import Dict, Any, List, Optional

import boto3, requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional Redis (recommended: Upstash). If absent, in-memory fallback is used.
try:
    import redis  # type: ignore
except Exception:
    redis = None  # noqa: F401

# Inference libs (serve locally from your R2 models)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ── ENV ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

ALLOW_ORIGINS       = os.getenv("ALLOW_ORIGINS", "*")

# Hugging Face hosted inference (optional fallback route #2)
HF_API_TOKEN        = os.getenv("HF_API_TOKEN")

# OpenAI fallback (optional fallback route #3)
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# RunPod serverless (GPU training)
RUNPOD_API_KEY      = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID  = os.getenv("RUNPOD_ENDPOINT_ID")

# R2 / S3 (to store trained adapters/weights)
S3_BUCKET           = os.getenv("S3_BUCKET")
S3_ENDPOINT         = os.getenv("S3_ENDPOINT")
AWS_REGION          = os.getenv("AWS_REGION", "auto")
AWS_ACCESS_KEY_ID   = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Redis (job tracking + model registry + per-model cfg)
REDIS_URL           = os.getenv("REDIS_URL", "")

# ── CLIENTS ──────────────────────────────────────────────────────────────────
s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

rdb = None
if REDIS_URL and redis is not None:
    try:
        rdb = redis.from_url(REDIS_URL)  # type: ignore
    except Exception:
        rdb = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/tmp/nspire_models"
os.makedirs(CACHE_DIR, exist_ok=True)

# HF base models you want to show users
PUBLIC_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # great for cheap tests
    "mistralai/Mistral-7B-v0.1",
    "tiiuae/falcon-7b",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-2.7B",
]

# RAM cache of loaded pipelines: model_id -> {pipe, meta}
local_models: Dict[str, Dict[str, Any]] = {}

# ── FASTAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(title="NSPIRE API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOW_ORIGINS] if ALLOW_ORIGINS != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SESSION (no login) ───────────────────────────────────────────────────────
def require_session(x_session_id: Optional[str] = Header(None)) -> str:
    """Every browser stores a UUID; we scope data by this id."""
    return x_session_id or f"anon-{uuid.uuid4()}"

# ── MODELS / SCHEMAS ─────────────────────────────────────────────────────────
class ModifyChat(BaseModel):
    model_id: str = Field(..., alias="modelId")
    temperature: float = 0.7
    token_limit: int = Field(256, alias="tokenLimit")
    instructions: str = Field("", alias="instructions")
    top_p: Optional[float] = Field(None, alias="topP")
    top_k: Optional[int]   = Field(None, alias="topK")
    stop: Optional[List[str]] = None
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

class RunChat(BaseModel):
    model_id: str = Field(..., alias="modelId")
    prompt: str
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

# ── REDIS KEYS / FALLBACK STORAGE ────────────────────────────────────────────
_mem_cfg: Dict[str, Dict[str, Any]] = {}
_mem_jobs: Dict[str, Dict[str, Any]] = {}
_mem_models: Dict[str, Dict[str, Any]] = {}

def _cfg_key(sid: str, mid: str) -> str: return f"cfg:{sid}:{mid}"
def _job_key(sid: str, jid: str) -> str: return f"job:{sid}:{jid}"
def _models_key(sid: str) -> str: return f"models:{sid}"

def _save_cfg(sid: str, mid: str, cfg: Dict[str, Any]):
    if rdb:
        rdb.hset(_cfg_key(sid, mid), mapping=cfg)
    else:
        _mem_cfg.setdefault(sid, {})[mid] = cfg

def _load_cfg(sid: str, mid: str) -> Dict[str, Any]:
    if rdb:
        raw = rdb.hgetall(_cfg_key(sid, mid))
        if not raw: return {}
        return {k.decode(): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
    return _mem_cfg.get(sid, {}).get(mid, {})

def _register_model(sid: str, mid: str, meta: Dict[str, Any]):
    if rdb:
        rdb.hset(_models_key(sid), mid, json.dumps(meta))
    else:
        _mem_models.setdefault(sid, {})[mid] = meta

def _list_models(sid: str) -> Dict[str, Any]:
    if rdb:
        h = rdb.hgetall(_models_key(sid))
        return {k.decode(): json.loads(v) for k, v in h.items()} if h else {}
    return _mem_models.get(sid, {})

def _save_job(sid: str, jid: str, mapping: Dict[str, Any]):
    if rdb:
        rdb.hset(_job_key(sid, jid), mapping=mapping)
    else:
        _mem_jobs.setdefault(sid, {}).setdefault(jid, {}).update(mapping)

def _get_job(sid: str, jid: str) -> Dict[str, Any]:
    if rdb:
        h = rdb.hgetall(_job_key(sid, jid))
        return {k.decode(): (v.decode() if isinstance(v, bytes) else v) for k, v in h.items()} if h else {}
    return _mem_jobs.get(sid, {}).get(jid, {})

# ── HELPERS ──────────────────────────────────────────────────────────────────
def _download_prefix_to(local_dir: str, bucket: str, prefix: str):
    os.makedirs(local_dir, exist_ok=True)
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token: kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            dest = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            s3.download_file(bucket, key, dest)
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break

def _ensure_loaded(model_id: str, meta: Dict[str, Any]):
    """Make sure a trained model (in S3) is on disk and live in a pipeline."""
    if model_id in local_models:
        return
    local_path = os.path.join(CACHE_DIR, model_id)
    if not os.path.isdir(local_path):
        # meta['s3_uri'] like s3://bucket/prefix
        uri = meta["s3_uri"]; assert uri.startswith("s3://")
        _, _, rest = uri.partition("s3://")
        bucket, _, prefix = rest.partition("/")
        _download_prefix_to(local_path, bucket, prefix)

    # load meta.json if present
    meta_path = os.path.join(local_path, "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
    base = meta.get("base_model_id")
    mtype = meta.get("type", "full")

    tok_id = base if base else local_path
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if mtype == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto" if DEVICE == "cuda" else None,
            load_in_4bit=(DEVICE == "cuda")
        )
        model = PeftModel.from_pretrained(base_model, local_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map="auto" if DEVICE == "cuda" else None
        )
    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=(0 if DEVICE == "cuda" else -1))
    local_models[model_id] = {"pipe": pipe, "meta": meta}

def _gen_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "max_new_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "do_sample": True,
        "return_full_text": False,
    }
    if cfg.get("top_p") not in ("", None): out["top_p"] = float(cfg["top_p"])
    if cfg.get("top_k") not in ("", None): out["top_k"] = int(cfg["top_k"])
    if cfg.get("stop"):
        out["stop_sequences"] = json.loads(cfg["stop"]) if isinstance(cfg["stop"], str) else cfg["stop"]
    return out

def _build_prompt(instr: str, u: str) -> str:
    i = (instr or "").strip()
    return f"{i}\n{u}" if i else u

def _rp(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/{path}"
    r = requests.post(url, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"ok": True, "device": DEVICE, "message": "NSPIRE API live"}

@app.get("/models")
def models(session_id: str = Depends(require_session)):
    return {"baseModels": PUBLIC_MODELS, "localModels": _list_models(session_id)}

@app.post("/modify-file")
def modify(req: ModifyChat, session_id: str = Depends(require_session)):
    cfg = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
        "top_p": req.top_p if req.top_p is not None else "",
        "top_k": req.top_k if req.top_k is not None else "",
        "stop": json.dumps(req.stop) if req.stop else "",
    }
    _save_cfg(session_id, req.model_id, cfg)
    return {"success": True, "modelId": req.model_id}

@app.post("/train")
async def train(
    base_model_id: str = Form(...),
    files: List[UploadFile] = File(...),
    use_lora: bool = Form(True),
    session_id: str = Depends(require_session)
):
    """
    Accepts uploaded files; we keep TEXT only (ignore images/binary).
    Sends a RunPod job to train on those texts; writes outputs to R2/S3.
    """
    texts: List[str] = []
    for f in files:
        # Filter to text-like content
        ct = (f.content_type or "").lower()
        if ct.startswith("image/"):  # ignore images for text LLM fine-tune
            continue
        b = await f.read()
        s = b.decode("utf-8", errors="ignore").strip()
        if s:
            texts.append(s)

    if not texts:
        raise HTTPException(400, "No valid text provided")

    if not (RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID):
        raise HTTPException(500, "RunPod not configured")

    job_id = str(uuid.uuid4())
    out_prefix = f"models/{session_id}/{job_id}"

    rp = _rp("run", {"input": {
        "base_model_id": base_model_id,
        "texts": texts,
        "use_lora": use_lora,
        "epochs": 1, "max_len": 256, "lr": 2e-4, "batch_size": 1,
        "out_prefix": out_prefix
    }})
    _save_job(session_id, job_id, {"rp_job_id": rp["id"], "status": "queued", "out_prefix": out_prefix})
    return {"job_id": job_id, "status": "queued"}

@app.get("/progress/{job_id}")
def progress(job_id: str, session_id: str = Depends(require_session)):
    job = _get_job(session_id, job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    rp_job_id = job.get("rp_job_id")
    r = requests.post(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"id": rp_job_id}, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    status = data.get("status")

    if status == "COMPLETED":
        out = data.get("output", {})
        model_id = f"ft-{job_id}"
        meta = {
            "s3_uri": out.get("s3_uri"),
            "type": out.get("type"),
            "base_model_id": out.get("base_model_id"),
            "created": int(time.time()),
        }
        _register_model(session_id, model_id, meta)
        _save_job(session_id, job_id, {"status": "completed", "model_id": model_id})
        return {"status": "completed", "model_id": model_id, "meta": meta}

    if status in ("IN_PROGRESS", "IN_QUEUE"):
        return {"status": "in_progress"}

    _save_job(session_id, job_id, {"status": "failed"})
    return {"status": "failed"}

@app.get("/load-local")
def load_local(model_id: str = Query(...), session_id: str = Depends(require_session)):
    models = _list_models(session_id)
    meta = models.get(model_id)
    if not meta:
        raise HTTPException(404, "Model not found for this session")
    _ensure_loaded(model_id, meta)
    return {"success": True, "loaded": model_id}

@app.post("/run")
def run_chat(req: RunChat, session_id: str = Depends(require_session)):
    cfg = _load_cfg(session_id, req.model_id)
    instr = cfg.get("instructions", "")

    # 1) Use trained local model if registered for this session
    meta = _list_models(session_id).get(req.model_id)
    if meta:
        _ensure_loaded(req.model_id, meta)
        pipe = local_models[req.model_id]["pipe"]
        out = pipe(_build_prompt(instr, req.prompt), **_gen_kwargs(cfg))
        return {"success": True, "source": "local", "response": out[0]["generated_text"].strip()}

    # 2) Hugging Face hosted inference (if modelId is a HF id)
    if HF_API_TOKEN:
        try:
            url = f"https://api-inference.huggingface.co/models/{req.model_id}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {
                "inputs": _build_prompt(instr, req.prompt),
                "parameters": {
                    "temperature": float(cfg.get("temperature", 0.7)),
                    "max_new_tokens": int(cfg.get("max_tokens", 256)),
                },
            }
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            else:
                text = str(data)
            return {"success": True, "source": "hf_inference", "response": text}
        except Exception as e:
            logging.warning(f"HF inference failed: {e}")

    # 3) OpenAI fallback
    if not OPENAI_API_KEY:
        raise HTTPException(502, "All backends unavailable")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages = []
        if instr:
            messages.append({"role": "system", "content": instr})
        messages.append({"role": "user", "content": req.prompt})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=float(cfg.get("temperature", 0.7)),
            max_tokens=int(cfg.get("max_tokens", 256)),
        )
        text = resp.choices[0].message.content.strip()
        return {"success": True, "source": "openai_fallback", "response": text}
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        raise HTTPException(502, "All backends failed")

# ── UVICORN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
