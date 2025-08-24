#!/usr/bin/env python3
import os, json, uuid, time, logging, tempfile, tarfile, shutil
from typing import Dict, Any, List, Optional, Tuple

import boto3, requests, redis
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging / Env
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)

ALLOW_ORIGINS         = os.getenv("ALLOW_ORIGINS", "*")
RUNPOD_API_KEY        = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID    = os.getenv("RUNPOD_ENDPOINT_ID")

S3_BUCKET             = os.getenv("S3_BUCKET")
S3_ENDPOINT           = os.getenv("S3_ENDPOINT")
AWS_REGION            = os.getenv("AWS_REGION", "auto")
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

REDIS_URL             = os.getenv("REDIS_URL", "")

HF_API_TOKEN          = os.getenv("HF_API_TOKEN")       # optional
OPENAI_API_KEY        = os.getenv("OPENAI_API_KEY")     # optional
FORCE_REMOTE          = os.getenv("FORCE_REMOTE", "0")  # "1" to skip local heavy loads

# ─────────────────────────────────────────────────────────────────────────────
# Clients / Caches
# ─────────────────────────────────────────────────────────────────────────────
s3 = None
if S3_BUCKET and S3_ENDPOINT and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )

rdb = redis.from_url(REDIS_URL) if REDIS_URL else None

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "/tmp/nspire_models"; os.makedirs(CACHE_DIR, exist_ok=True)

PUBLIC_MODELS = [
    # pick repos that respond on HF Inference without gating
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "distilbert/distilbert-base-uncased",  # tokenizers etc (not for chat)
]

# Local loaded models cache
local_models: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI + CORS
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
# Session helper (header OR query ?sid=)
# ─────────────────────────────────────────────────────────────────────────────
def require_session(
    x_session_id: Optional[str] = Header(None),
    sid: Optional[str] = Query(None)
) -> str:
    return x_session_id or sid or f"anon-{uuid.uuid4()}"

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
# Redis helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cfg_key(sid: str, mid: str) -> str:  return f"cfg:{sid}:{mid}"
def _job_key(sid: str, jid: str) -> str:  return f"job:{sid}:{jid}"
def _models_key(sid: str) -> str:         return f"models:{sid}"

def _save_cfg(sid: str, mid: str, cfg: Dict[str,Any]):
    if rdb: rdb.hset(_cfg_key(sid,mid), mapping=cfg)

def _load_cfg(sid: str, mid: str) -> Dict[str,Any]:
    if not rdb: return {}
    raw = rdb.hgetall(_cfg_key(sid,mid))
    if not raw: return {}
    out = {k.decode(): v.decode() if isinstance(v,bytes) else v for k,v in raw.items()}
    # normalize types
    if "temperature" in out: out["temperature"] = float(out["temperature"])
    if "max_tokens"  in out: out["max_tokens"]  = int(out["max_tokens"])
    return out

def _register_model(sid: str, mid: str, meta: Dict[str,Any]):
    if rdb: rdb.hset(_models_key(sid), mid, json.dumps(meta))

def _list_models(sid: str) -> Dict[str,Any]:
    if not rdb: return {}
    h = rdb.hgetall(_models_key(sid))
    return {k.decode(): json.loads(v) for k,v in h.items()} if h else {}

# ─────────────────────────────────────────────────────────────────────────────
# S3 helpers
# ─────────────────────────────────────────────────────────────────────────────
def _parse_s3_uri(uri: str) -> Tuple[str,str]:
    assert uri.startswith("s3://"), "s3_uri must start with s3://"
    _,_,rest = uri.partition("s3://")
    bucket,_,prefix = rest.partition("/")
    return bucket, prefix

def _download_prefix(local_dir: str, bucket: str, prefix: str):
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

def _list_prefix(bucket: str, prefix: str) -> List[Dict[str,Any]]:
    out = []
    token=None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token: kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            out.append(obj)
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Model load & inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_loaded(model_id: str, meta: Dict[str,Any]):
    if FORCE_REMOTE == "1":
        # caller wants to avoid local heavy loads (cheap CPU plan etc.)
        raise RuntimeError("FORCE_REMOTE=1 prohibits local load")
    if model_id in local_models: 
        return

    # folder on disk
    local_path = os.path.join(CACHE_DIR, model_id)
    if not os.path.isdir(local_path):
        uri = meta["s3_uri"]
        bucket, prefix = _parse_s3_uri(uri)
        if not s3:
            raise RuntimeError("S3 not configured")
        logging.info(f"Downloading trained model from s3://{bucket}/{prefix} …")
        _download_prefix(local_path, bucket, prefix)

    # (optional) override meta with on-disk meta.json
    meta_path = os.path.join(local_path,"meta.json")
    if os.path.exists(meta_path):
        try:
            meta.update(json.load(open(meta_path)))
        except: 
            pass

    base = meta.get("base_model_id")
    mtype = meta.get("type","full")

    tok_id = base if base else local_path
    tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if mtype == "lora":
        base_model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto" if DEVICE=="cuda" else None,
            load_in_4bit=(DEVICE=="cuda")
        )
        model = PeftModel.from_pretrained(base_model, local_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            local_path, device_map="auto" if DEVICE=="cuda" else None
        )

    model.eval()
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device=(0 if DEVICE=="cuda" else -1))
    local_models[model_id] = {"pipe": pipe, "meta": meta}
    logging.info(f"Loaded local model: {model_id}")

def _gen_kwargs(cfg: Dict[str,Any]) -> Dict[str,Any]:
    out = {
        "max_new_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.7)),
        "do_sample": True,
        "return_full_text": False
    }
    if cfg.get("top_p") not in ("", None): out["top_p"] = float(cfg["top_p"])
    if cfg.get("top_k") not in ("", None): out["top_k"] = int(cfg["top_k"])
    if cfg.get("stop"):
        out["stop_sequences"] = json.loads(cfg["stop"]) if isinstance(cfg["stop"], str) else cfg["stop"]
    return out

def _build_prompt(instr: str, user: str) -> str:
    i = (instr or "").strip()
    return f"{i}\n{user}" if i else user

def _prewarm_model(mid: str, cfg: Dict[str,Any], sid: str):
    """
    If mid is fine-tuned (ft-...), load locally from S3.
    Otherwise, ping HF Inference so the first real call is faster.
    """
    # If it's an FT model (registered for this session), load it 
    if mid.startswith("ft-"):
        meta = _list_models(sid).get(mid)
        if meta:
            try:
                _ensure_loaded(mid, meta)
                return "local"
            except Exception as e:
                logging.warning(f"Local prewarm failed: {e}")

    if HF_API_TOKEN:
        try:
            url = f"https://api-inference.huggingface.co/models/{mid}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {
                "inputs": _build_prompt(cfg.get("instructions",""), "hello"),
                "parameters": {"max_new_tokens": 8}
            }
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            # 503 usually means "loading", which is fine for prewarm
            if r.status_code == 404:
                logging.warning(f"HF 404 for repo {mid} — repo ID / gating / token access.")
            return "hf_inference"
        except Exception as e:
            logging.warning(f"HF prewarm failed: {e}")

    return "none"

# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"ok": True, "device": DEVICE, "message": "nspire API live"}

@app.get("/models")
def models(session_id: str = Depends(require_session)):
    return {"baseModels": PUBLIC_MODELS, "localModels": _list_models(session_id)}

@app.get("/config")
def get_cfg(model_id: str = Query(...), session_id: str = Depends(require_session)):
    return _load_cfg(session_id, model_id)

@app.post("/modify-file")
def modify(
    req: ModifyChat, 
    prewarm: int = Query(0), 
    session_id: str = Depends(require_session)
):
    cfg = {
        "temperature": req.temperature,
        "max_tokens": req.token_limit,
        "instructions": req.instructions,
        "top_p": req.top_p if req.top_p is not None else "",
        "top_k": req.top_k if req.top_k is not None else "",
        "stop": json.dumps(req.stop) if req.stop else ""
    }
    _save_cfg(session_id, req.model_id, cfg)

    warmed = "none"
    if prewarm:
        warmed = _prewarm_model(req.model_id, cfg, session_id)

    return {"success": True, "modelId": req.model_id, "prewarmed": warmed}

# --- Stub train -> RunPod (leave as you had; omitted here for brevity) ---
def _rp(path: str, payload: Dict[str,Any]) -> Dict[str,Any]:
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise HTTPException(500, "RunPod not configured")
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/{path}"
    r = requests.post(url, headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

@app.post("/train")
async def train(
    base_model_id: str = Form(...),
    files: List[UploadFile] = File(...),
    use_lora: bool = Form(True),
    session_id: str = Depends(require_session)
):
    texts=[]
    for f in files:
        b=await f.read()
        s=b.decode("utf-8", errors="ignore").strip()
        if s: texts.append(s)
    if not texts:
        raise HTTPException(400, "No valid text provided")

    job_id = str(uuid.uuid4())
    out_prefix = f"models/{session_id}/{job_id}"

    rp = _rp("run", {"input": {
        "base_model_id": base_model_id,
        "texts": texts,
        "use_lora": use_lora,
        "epochs": 1, "max_len": 256, "lr": 2e-4, "batch_size": 1,
        "out_prefix": out_prefix
    }})
    if rdb:
        rdb.hset(_job_key(session_id, job_id), mapping={
            "rp_job_id": rp["id"], "status": "queued", "out_prefix": out_prefix
        })
    return {"job_id": job_id, "status": "queued"}

@app.get("/progress/{job_id}")
def progress(job_id: str, session_id: str = Depends(require_session)):
    if not rdb:
        raise HTTPException(500, "Redis required")
    h = rdb.hgetall(_job_key(session_id, job_id))
    if not h:
        raise HTTPException(404, "Job not found")
    rp_id = h[b"rp_job_id"].decode()

    r = requests.post(
        f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        json={"id": rp_id}, timeout=30
    )
    r.raise_for_status()
    data = r.json()
    status = data.get("status")

    if status == "COMPLETED":
        out = data.get("output", {})
        model_id = f"ft-{job_id}"
        meta = {
            "s3_uri": out.get("s3_uri"),
            "type": out.get("type", "lora"),
            "base_model_id": out.get("base_model_id"),
            "created": int(time.time())
        }
        _register_model(session_id, model_id, meta)
        rdb.hset(_job_key(session_id, job_id), mapping={"status":"completed","model_id":model_id})
        return {"status":"completed","model_id":model_id,"meta":meta}
    elif status in ("IN_PROGRESS","IN_QUEUE"):
        return {"status":"in_progress"}
    else:
        rdb.hset(_job_key(session_id, job_id), mapping={"status":"failed"})
        return {"status":"failed"}

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
    instr = cfg.get("instructions","")

    # 1) try local FT model
    meta = _list_models(session_id).get(req.model_id)
    if meta and FORCE_REMOTE != "1":
        try:
            _ensure_loaded(req.model_id, meta)
            pipe = local_models[req.model_id]["pipe"]
            out = pipe(_build_prompt(instr, req.prompt), **_gen_kwargs(cfg))
            return {"success": True, "response": out[0]["generated_text"].strip()}
        except Exception as e:
            logging.warning(f"Local inference failed, falling back: {e}")

    # 2) HF hosted inference (public)
    if HF_API_TOKEN:
        try:
            url = f"https://api-inference.huggingface.co/models/{req.model_id}"
            headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
            payload = {"inputs": _build_prompt(instr, req.prompt),
                       "parameters": {"temperature": float(cfg.get("temperature",0.7)),
                                      "max_new_tokens": int(cfg.get("max_tokens",256))}}
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 404:
                logging.warning(f"HF 404 for repo {req.model_id} — repo ID / gating / token access.")
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            else:
                text = str(data)
            return {"success": True, "response": text}
        except Exception as e:
            logging.warning(f"HF inference failed: {e}")

    # 3) OpenAI fallback
    if not OPENAI_API_KEY:
        raise HTTPException(502, "No remote backend available. Set HF_API_TOKEN or OPENAI_API_KEY.")
    try:
        # openai>=1.0
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        messages=[]
        if instr: messages.append({"role":"system","content":instr})
        messages.append({"role":"user","content":req.prompt})
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=float(cfg.get("temperature",0.7)),
            max_tokens=int(cfg.get("max_tokens",256))
        )
        text = resp.choices[0].message.content.strip()
        return {"success": True, "response": text}
    except Exception as e:
        logging.error(f"OpenAI fallback failed: {e}")
        raise HTTPException(502, "All backends unavailable")

# ─────────────────────────────────────────────────────────────────────────────
# Downloads
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/download_manifest")
def download_manifest(model_id: str = Query(...), session_id: str = Depends(require_session)):
    if not s3:
        raise HTTPException(500, "S3 not configured")
    meta = _list_models(session_id).get(model_id)
    if not meta or "s3_uri" not in meta:
        raise HTTPException(404, "Model not found or not downloadable")
    bucket, prefix = _parse_s3_uri(meta["s3_uri"])
    objs = _list_prefix(bucket, prefix)
    files = []
    for o in objs:
        key = o["Key"]
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=1800,  # 30 min
        )
        files.append({"key": key, "size": o.get("Size", 0), "url": url})
    return {"model_id": model_id, "count": len(files), "files": files}

@app.get("/download_zip")
def download_zip(model_id: str = Query(...), session_id: str = Depends(require_session)):
    if not s3:
        raise HTTPException(500, "S3 not configured")
    meta = _list_models(session_id).get(model_id)
    if not meta or "s3_uri" not in meta:
        raise HTTPException(404, "Model not found or not downloadable")
    bucket, prefix = _parse_s3_uri(meta["s3_uri"])

    tmp_dir = tempfile.mkdtemp(prefix=f"{model_id}_")
    try:
        _download_prefix(tmp_dir, bucket, prefix)
        # build a tar.gz
        archive = os.path.join("/tmp", f"{model_id}.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(tmp_dir, arcname=model_id)
        return FileResponse(archive, media_type="application/gzip", filename=f"{model_id}.tar.gz")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ─────────────────────────────────────────────────────────────────────────────
# Uvicorn
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT","8080")))
