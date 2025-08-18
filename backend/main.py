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
