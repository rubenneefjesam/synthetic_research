# secure fallback: prefer GROQ_API (Codespaces) and fallback to OPENAI_API_KEY
N = 10
import os

if os.getenv("GROQ_API") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API")

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
