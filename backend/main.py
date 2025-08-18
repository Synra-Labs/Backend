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
