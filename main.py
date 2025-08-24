#!/usr/bin/env python3
import os, json, uuid, time, logging
from typing import Dict, Any, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps (only used if present)
try:
    import boto3  # for S3/R2
except Exception:
    boto3 = None
try:
    import redis
except Exception:
    redis = None

# ─────────────────────────────────────────────────────────────────────────────
# Logging & env
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nspire")

ALLOW_ORIGINS         = os.getenv("ALLOW_ORIGINS", "*")
RUNPOD_API_KEY        = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID    = os.getenv("RUNPOD_ENDPOINT_ID")

S3_BUCKET             = os.getenv("S3_BUCKET")
S3_ENDPOINT           = os.getenv("S3_ENDPOINT")
AWS_REGION            = os.getenv("AWS_REGION", "auto")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

REDIS_URL             = os.getenv("REDIS_URL", "")

HF_API_TOKEN          = os.getenv("HF_API_TOKEN")        # optional
HF_INFERENCE_FALLBACK = os.getenv("HF_INFERENCE_FALLBACK", "HuggingFaceH4/zephyr-7b-beta")

OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")      # optional
OPENAI_CHAT_MODEL     = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# ─────────────────────────────────────────────────────────────────────────────
# Optional clients
# ─────────────────────────────────────────────────────────────────────────────
s3 = None
if boto3 and S3_BUCKET and S3_ENDPOINT and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
        log.info("S3 client ready.")
    except Exception as e:
        log.warning(f"S3 disabled: {e}")
        s3 = None

rdb = None
if redis and REDIS_URL:
    try:
        rdb = redis.from_url(REDIS_URL)
        rdb.ping()
        log.info("Redis connected.")
    except Exception as e:
        log.warning(f"Redis disabled: {e}")
        rdb = None

# in-process fallback store (used when Redis not set)
_mem_cfg: Dict[str, Dict[str, Any]] = {}
_mem_models: Dict[str, Dict[str, Any]] = {}
_mem_jobs: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI & CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="nspire API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOW_ORIGINS == "*" else [ALLOW_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Session (no login — per-browser header)
# ─────────────────────────────────────────────────────────────────────────────
def require_session(x_session_id: Optional[str] = Header(None)) -> str:
    return x_session_id or f"anon-{uuid.uuid4()}"

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Storage helpers (Redis or in-memory)
# ─────────────────────────────────────────────────────────────────────────────
def _cfg_key(sid: str, mid: str) -> str:  return f"cfg:{sid}:{mid}"
def _job_key(sid: str, jid: str) -> str:  return f"job:{sid}:{jid}"
def _models_key(sid: str) -> str:         return f"models:{sid}"

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

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PUBLIC_MODELS = [
    # Use HF repos that are typically available for serverless or well-known
    "HuggingFaceH4/zephyr-7b-beta",
    "google/gemma-2-2b-it",
    "NousResearch/Hermes-2-Pro-Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "tiiuae/falcon-7b-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # may not be on serverless; fallback handles it
]

# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_prompt(instr: str, user: str) -> str:
    i = (instr or "").strip()
    return f"{i}\n{user}" if i else user

def _gen_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "max_new_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.7)),
    }
    if cfg.get("top_p") not in ("", None): out["top_p"] = float(cfg["top_p"])
    if cfg.get("top_k") not in ("", None): out["top_k"] = int(cfg["top_k"])
    return out

def _hf_infer(repo_id: str, full_input: str, params: Dict[str, Any]) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN not set")

    url = f"https://api-inference.huggingface.co/models/{repo_id}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": full_input,
        "parameters": params,
        "options": {"wait_for_model": True},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code == 404:
        raise FileNotFoundError(f"HF 404 for repo {repo_id}")
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    # Some endpoints return {"conversation": {"generated_responses":[...]}}
    try:
        conv = data.get("conversation", {})
        gen = conv.get("generated_responses", [])
        if gen:
            return gen[-1]
    except Exception:
        pass
    return str(data)

def _openai_chat(full_input: str, cfg: Dict[str, Any]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "user", "content": full_input}]
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=float(cfg.get("temperature", 0.7)),
        max_tokens=int(cfg.get("max_tokens", 256)),
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"ok": True, "message": "nspire API live", "hf": bool(HF_API_TOKEN), "openai": bool(OPENAI_API_KEY)}

@app.get("/models")
def models(session_id: str = Depends(require_session)):
    return {"baseModels": PUBLIC_MODELS, "localModels": _list_models(session_id)}

@app.get("/config")
def get_config(model_id: str, session_id: str = Depends(require_session)):
    return _load_cfg(session_id, model_id)

@app.post("/modify-file")
def modify(req: ModifyChat, session_id: str = Depends(require_session), prewarm: int = 0):
    cfg = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
        "top_p": "" if req.top_p is None else req.top_p,
        "top_k": "" if req.top_k is None else req.top_k,
        "stop": json.dumps(req.stop) if req.stop else "",
    }
    _save_cfg(session_id, req.model_id, cfg)
    return {"success": True, "modelId": req.model_id, "prewarmed": bool(prewarm)}

# (Train/progress endpoints omitted here for brevity – keep your existing ones if you’re using Runpod)

@app.post("/run")
def run_chat(req: RunChat, session_id: str = Depends(require_session)):
    cfg  = _load_cfg(session_id, req.model_id)
    instr = cfg.get("instructions", "")
    full_input = _build_prompt(instr, req.prompt)
    params = _gen_params(cfg)

    # 1) Try HF serverless
    if HF_API_TOKEN:
        repo = req.model_id
        try:
            text = _hf_infer(repo, full_input, params)
            return {"success": True, "source": f"hf:{repo}", "response": text}
        except FileNotFoundError:
            log.warning(f"HF 404 for repo {repo} — trying fallback {HF_INFERENCE_FALLBACK}")
            try:
                text = _hf_infer(HF_INFERENCE_FALLBACK, full_input, params)
                return {"success": True, "source": f"hf:{HF_INFERENCE_FALLBACK}", "response": text}
            except Exception as e2:
                log.warning(f"HF inference failed (fallback): {e2}")
        except Exception as e:
            log.warning(f"HF inference failed: {e}")

    # 2) OpenAI fallback (v1 only)
    if OPENAI_API_KEY:
        try:
            text = _openai_chat(full_input, cfg)
            return {"success": True, "source": f"openai:{OPENAI_CHAT_MODEL}", "response": text}
        except Exception as e:
            log.error(f"OpenAI v1 failed: {e}")

    # 3) Nothing worked
    raise HTTPException(
        status_code=502,
        detail="No remote backend available. Set HF_API_TOKEN with access to the repo or OPENAI_API_KEY.",
    )

# Optional: simple download manifest/zip endpoints for your Downloads page.
@app.get("/download_manifest")
def download_manifest(model_id: str, sid: Optional[str] = Query(None)):
    # This sample assumes your fine-tunes are archived in S3/R2 under models/<sid>/<job>/...
    # Implement your own mapping. If no S3 configured, return empty.
    if not s3:
        return {"files": []}
    # Example placeholder; customize to your path layout.
    prefix = f"models/{sid}/{model_id}/"
    try:
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        files = []
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            head = s3.head_object(Bucket=S3_BUCKET, Key=key)
            size = int(head.get("ContentLength", 0))
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": key},
                ExpiresIn=3600,
            )
            files.append({"key": key, "url": url, "size": size})
        return {"files": files}
    except Exception as e:
        log.warning(f"download_manifest error: {e}")
        return {"files": []}

@app.get("/download_zip")
def download_zip(model_id: str, sid: Optional[str] = Query(None)):
    # For large sets, you should stream a server-side zip/tar; here we just signal.
    if not s3:
        raise HTTPException(404, "No storage configured")
    return {"message": "Implement streaming/tarball here for your storage layout."}

# ─────────────────────────────────────────────────────────────────────────────
# Uvicorn entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
