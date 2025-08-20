#!/usr/bin/env python3
"""
Ailo Forge Backend (main.py)

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
