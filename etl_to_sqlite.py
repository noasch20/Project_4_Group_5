#!/usr/bin/env python3
"""
etl_to_sqlite.py
----------------
Load the Kaggle Disease–Symptom dataset, apply the same cleaning steps you use
for modelling, and persist the result to a local SQLite database
(`disease_symptoms.db`).

Run:
    python etl_to_sqlite.py \
        --train-csv data/Training.csv \
        --db-file disease_symptoms.db \
        --table-name symptoms
"""

from pathlib import Path
import argparse
import pandas as pd
from sqlalchemy import create_engine


# ---------------------------------------------------------------------------
# 0. CLI ─────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True,
                   help="Path to Training.csv from Kaggle")
    p.add_argument("--db-file", type=Path, default=Path("disease_symptoms.db"),
                   help="SQLite file to create or overwrite")
    p.add_argument("--table-name", type=str, default="symptoms",
                   help="Destination table name inside SQLite")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 1. Load & CLEAN ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # --- SAMPLE cleaning steps (adapt/extend to match your modeller) --------
    df.columns = (
        df.columns.str.strip()          # remove stray spaces
                 .str.lower()           # lower-case for consistency
                 .str.replace(" ", "_") # spaces → underscores
    )

    # Ensure symptom flags are 0/1 integers, not strings or NaNs
    symptom_cols = df.columns.difference(["prognosis"])
    df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    return df


# ---------------------------------------------------------------------------
# 2. Write to SQLite ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
def write_sqlite(df: pd.DataFrame, db_file: Path, table: str):
    engine = create_engine(f"sqlite:///{db_file}")
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"✅  Wrote {len(df):,} rows → {db_file} (table = '{table}')")


# ---------------------------------------------------------------------------
# 3. Entrypoint ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    df_clean = load_and_clean(args.train_csv)
    write_sqlite(df_clean, args.db_file, args.table_name)
