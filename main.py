#!/usr/bin/env python3
"""
Synra Nspire Backend (main.py)

• /models        → list available public HF models
• /modify-file   → set (in-memory) chat parameters per model
• /run           → generate text via HF Inference API, fallback to OpenAI GPT-4
• /train         → stubbed fine-tune (with progress polling)
"""
import os
import logging
import uuid
import requests
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai import OpenAI
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# 1) ENV + CLIENT SETUP
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if HF_API_TOKEN:
    logging.info("✅ HF_API_TOKEN loaded")
else:
    logging.warning("⚠️ HF_API_TOKEN not set — HF inference disabled")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logging.error("❌ OPENAI_API_KEY is required for GPT-4 fallback")
    raise RuntimeError("Missing OPENAI_API_KEY")
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# in-memory stores
chat_cfg: Dict[str, Dict[str, Any]] = {}
train_progress: Dict[str, Dict[str, Any]] = {}

# public models (no gated repos)
PUBLIC_MODELS = [
    "mistralai/mistral-7b",
    "tiiuae/falcon-40b",
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neo-2.7B",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2) FASTAPI SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Ailo Forge", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ─────────────────────────────────────────────────────────────────────────────
# 3) SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
class ModifyChat(BaseModel):
    model_id:    str   = Field(..., alias="modelId")
    temperature: float
    token_limit: int   = Field(..., alias="tokenLimit")
    instructions:str   = Field("", alias="instructions")
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

class RunChat(BaseModel):
    model_id: str   = Field(..., alias="modelId")
    prompt:   str
    class Config:
        allow_population_by_alias = True
        allow_population_by_field_name = True

# ─────────────────────────────────────────────────────────────────────────────
# 4) ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def healthcheck():
    return {"status":"ok","message":"Ailo Forge backend is live"}

@app.get("/models")
async def list_models():
    return {"models": PUBLIC_MODELS}

@app.post("/modify-file")
async def modify_file(req: ModifyChat):
    chat_cfg[req.model_id] = {
        "temperature": req.temperature,
        "max_tokens":  req.token_limit,
        "instructions": req.instructions,
    }
    return {"success": True, "message": "✅ Model has been modified!"}

@app.post("/run")
async def run_chat(req: RunChat):
    cfg = chat_cfg.get(req.model_id, {})
    temp    = cfg.get("temperature", 0.7)
    max_tok = cfg.get("max_tokens", 150)
    instr   = cfg.get("instructions", "")

    # build full prompt
    full_input = instr.strip()
    if full_input:
        full_input += "\n" + req.prompt
    else:
        full_input = req.prompt

    # 1) HF Inference
    if HF_API_TOKEN:
        hf_url = f"https://api-inference.huggingface.co/models/{req.model_id}"
        headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        payload = {
            "inputs": full_input,
            "parameters": {"temperature": temp, "max_new_tokens": max_tok},
        }
        try:
            r = requests.post(hf_url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            # extract generated_text
            if isinstance(data, list) and "generated_text" in data[0]:
                text = data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                text = data["generated_text"]
            else:
                text = data.get("generated_text", str(data))
            return {"success": True, "response": text}
        except Exception as e:
            logging.warning(f"HF Inference error for {req.model_id}: {e} — falling back to OpenAI")

    # 2) OpenAI fallback
    messages = []
    if instr:
        messages.append({"role":"system","content":instr})
    messages.append({"role":"user","content":req.prompt})
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
        )
        text = resp.choices[0].message.content.strip()
        return {"success": True, "response": text}
    except Exception as e:
        logging.error(f"OpenAI GPT-4 error: {e}")
        raise HTTPException(502, detail="Both HF and OpenAI calls failed.")

@app.post("/train")
async def train_model(
    repo_id: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    texts = []
    for f in files:
        b = await f.read()
        if not b.startswith((b"\xFF\xD8", b"\x89PNG")):
            texts.append(b.decode("utf-8", errors="ignore"))
    if not texts:
        raise HTTPException(400, detail="No valid text provided")

    job = str(uuid.uuid4())
    train_progress[job] = {"percent": 0, "status": "in_progress"}
    background_tasks.add_task(_run_training, job, repo_id, texts)
    return {"job_id": job, "status": "training_started"}

@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in train_progress:
        raise HTTPException(404, detail="Job not found")
    return train_progress[job_id]

# ─────────────────────────────────────────────────────────────────────────────
# 5) BACKGROUND TRAINING STUB
# ─────────────────────────────────────────────────────────────────────────────
def _run_training(job_id: str, repo_id: str, texts: List[str]):
    try:
        tok = AutoTokenizer.from_pretrained(repo_id)
        mod = AutoModelForCausalLM.from_pretrained(repo_id)
        ds  = Dataset.from_dict({"text": texts})
        ds  = ds.map(lambda x: tok(x["text"], truncation=True, max_length=128), batched=True)

        out_dir = f"models/ft-{job_id}"
        args = TrainingArguments(output_dir=out_dir, num_train_epochs=3, per_device_train_batch_size=2, push_to_hub=False)
        tr = Trainer(mod, args, train_dataset=ds, tokenizer=tok)
        tr.train(); tr.save_model(out_dir)

        train_progress[job_id] = {"percent": 100, "status": "completed"}
    except:
        logging.exception("Training failed")
        train_progress[job_id] = {"percent": 0, "status": "failed"}

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
