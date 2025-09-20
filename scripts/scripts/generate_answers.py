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
import openai
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # pas aan als nodig

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
        werkervaring_jaren=int(row.get("werkervaring_jaren", 0)),
        senioriteit=row.get("senioriteit", ""),
        ai_affiniteit=row.get("ai_affiniteit", "Gemiddeld"),
        pct_pc=row.get("%_werk_op_computer", row.get("pct_pc", 50)),
    )

def call_llm(prompt: str):
    try:
        resp = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Je bent een beknopte, praktische adviseur. Antwoord in het Nederlands."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        return text, None
    except Exception as e:
        return None, str(e)

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
    for i, row in enumerate(tqdm(sample_df.itertuples(index=False), total=len(sample_df), desc="generating")):
        # convert namedtuple to dict-like access
        rowd = row._asdict() if hasattr(row, "_asdict") else row._asdict()
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
