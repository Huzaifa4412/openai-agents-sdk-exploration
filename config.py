import os
from dotenv import load_dotenv
from pathlib import Path


# ✅ Root directory detect
ROOT_DIR = Path(__file__).resolve().parent
# print(ROOT_DIR)

load_dotenv(ROOT_DIR / ".env")


# ✅ API Keys (add more if needed)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini Api key not found ")
