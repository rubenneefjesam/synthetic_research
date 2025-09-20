#!/usr/bin/env python3
# scripts/generate_answers.py
# Minimal, robust generator for testing (N=10 default).
# Uses OPENAI_API_KEY (fallback from GROQ_API if present).

import os
import json
import time
import random
from pathlib import Path

# --- env fallback (Codespaces) ---
if os.getenv("GROQ_API") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API")

import pandas as pd
from tqdm import tqdm

# Note: we will use the new openai client when calling the LLM inside call_llm
# openai.api_key is kept for backward compat if needed by other libs
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # adjust if necessary

# --- Parameters (test) ---
N = 10  # test-run; zet op 1000 voor full-run
CSV_PATH = Path("data/volker_nexus_group_populatie_2000_clean.csv")
OUT_PATH = Path("results/generated_answers.jsonl")
SEED = 42
USER_QUESTION = (
    "Wat moet Volker Nexus Group doen om AI verantwoord en snel te integreren in "
    "werkvoorbereiding en uitvoering? Geef (A) één korte principiële aanbeveling, "
    "(B) twee concrete acties die deze medewerker kan ondersteunen of voorstellen, "
    "en (C) één risico/bezwaar dat deze persoon waarschijnlijk voelt."
)

PROMPT_TEMPLATE = """
Je bent een collega-adviseur bij Volker Nexus Group (bouw/infra).
Hieronder staan kenmerken van één medewerker. Schrijf in maximaal ~120-200 woorden een korte, op deze persoon toegespitste reactie op de vraag:
Vraag: {question}

Medewerker (kort):
- Functie: {functie}
- Afdeling: {afdeling}
- Team: {team}
- Studie: {studie_niveau} ({opleidingsrichting})
- Specialisatie: {specialisatie}
- Werkervaring (jaren): {werkervaring_jaren}
- Senioriteit: {senioriteit}
- AI-affiniteit: {ai_affiniteit}
- % werk op computer: {pct_pc}

Antwoordformaat (verplicht):
A) Eén korte principiële aanbeveling (1 zin).
B) Twee concrete acties (elk 1 regel) die deze persoon kan ondersteunen of voorstellen.
C) Eén risico/bezwaar dat deze persoon waarschijnlijk voelt (1 zin).
Wees bondig, praktisch, Nederlands, max 200 woorden totaal.
"""

random.seed(SEED)

def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"CSV niet gevonden: {path}")
    df = pd.read_csv(path)
    return df

def build_prompt(row):
    return PROMPT_TEMPLATE.format(
        question=USER_QUESTION,
        functie=row.get("functie", ""),
        afdeling=row.get("afdeling", ""),
        team=row.get("team", ""),
        studie_niveau=row.get("studie_niveau", ""),
        opleidingsrichting=row.get("opleidingsrichting", row.get("opleiding","")),
        specialisatie=row.get("specialisatie", ""),
        werkervaring_jaren=int(row.get("werkervaring_jaren", 0) or 0),
        senioriteit=row.get("senioriteit", ""),
        ai_affiniteit=row.get("ai_affiniteit", "Gemiddeld"),
        pct_pc=row.get("%_werk_op_computer", row.get("pct_pc", 50)),
    )

def call_llm(prompt: str):
    """
    Groq HTTP caller. Tries a short list of Groq models (returns first successful).
    Uses env var GROQ_API. Returns (text, error).
    """
    import requests, json, os, time

    key = os.getenv("GROQ_API") or os.getenv("GROQ_API_KEY")
    if not key:
        return None, "No GROQ_API found in env"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # candidates (prefer explicit MODEL env, else try these)
    candidates = []
    env_model = os.getenv("MODEL")
    if env_model:
        candidates.append(env_model)
    candidates += [
        "groq/compound",
        "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b"
    ]

    last_err = None
    for m in candidates:
        payload = {
            "model": m,
            "messages": [
                {"role": "system", "content": "Je bent een beknopte, praktische adviseur. Antwoord in het Nederlands."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.3,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
        except Exception as e:
            last_err = f"Network error while trying model {m}: {e}"
            continue

        if resp.status_code == 200:
            try:
                j = resp.json()
                # prefer structured OpenAI-like response
                text = j["choices"][0]["message"]["content"]
            except Exception:
                # fallback: stringify JSON if structure differs
                text = json.dumps(j, ensure_ascii=False)
            return text.strip(), None
        else:
            last_err = f"Model {m} returned {resp.status_code}: {resp.text}"
            # for auth errors stop early
            if resp.status_code in (401, 403):
                return None, last_err
            # otherwise try next model
            time.sleep(0.2)

    return None, last_err or "No model succeeded"

def main():
    print("Start generator — CSV:", CSV_PATH)
    df = load_data(CSV_PATH)
    if len(df) < N:
        print(f"Waarschuwing: CSV kleiner ({len(df)}) dan N={N}. Gebruik alle rijen.")
        sample_df = df.copy()
    else:
        sample_df = df.sample(N, random_state=SEED).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(OUT_PATH, "w", encoding="utf-8")

    errors = 0
    for i in range(len(sample_df)):
        row_series = sample_df.iloc[i]
        rowd = row_series.to_dict()
        prompt = build_prompt(rowd)
        print(f"\n[{i+1}/{len(sample_df)}] Generating for medewerker_id={rowd.get('medewerker_id','?')} functie={rowd.get('functie','')}")
        text, err = call_llm(prompt)
        if err:
            errors += 1
            print(f"  ERROR: {err}")
            out = {
                "idx": i,
                "medewerker_id": rowd.get("medewerker_id"),
                "functie": rowd.get("functie"),
                "answer": None,
                "error": err
            }
        else:
            out = {
                "idx": i,
                "medewerker_id": rowd.get("medewerker_id"),
                "functie": rowd.get("functie"),
                "answer": text
            }
            print("  OK — length:", len(text))
        out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        out_f.flush()
        time.sleep(0.25)  # rate-limit friendly

    out_f.close()
    print(f"Done. Wrote results to {OUT_PATH}. Errors: {errors}")

if __name__ == "__main__":
    main()
