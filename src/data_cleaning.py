# src/data_cleaning.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sqlalchemy import text
from database.db_connections import get_engine

def load_raw_data(path="data/IPL.csv"):
    """Read the raw IPL dataset"""
    print("📥 Loading dataset ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def clean_matches_data(df: pd.DataFrame):
    """Select and clean match-level columns for insertion into MySQL"""
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Pick only columns that exist in this dataset
    match_cols = [
        "id", "season", "date", "city", "venue", "team1", "team2",
        "toss_winner", "toss_decision", "result", "dl_applied",
        "winner", "win_by_runs", "win_by_wickets", "player_of_match",
        "umpire1", "umpire2"
    ]
    available_cols = [c for c in match_cols if c in df.columns]
    m = df[available_cols].copy()

    # Rename 'id' → 'match_id'
    if "id" in m.columns:
        m.rename(columns={"id": "match_id"}, inplace=True)

    # Convert and clean date
    if "date" in m.columns:
        m["match_date"] = pd.to_datetime(m["date"], errors="coerce")
        m.drop(columns=["date"], inplace=True)

    # Handle numeric columns dynamically
    numeric_cols = [c for c in ["season", "dl_applied", "win_by_runs", "win_by_wickets"] if c in m.columns]
    for col in numeric_cols:
        if col == "season":
            # Handle multi-year values like 2007/08
            def normalize_season(val):
                if pd.isna(val):
                    return 0
                val = str(val)
                if "/" in val:
                    parts = val.split("/")
                    try:
                        return int("20" + parts[1]) if len(parts[1]) == 2 else int(parts[1])
                    except:
                        return 0
                try:
                    return int(val)
                except:
                    return 0
            m["season"] = m["season"].apply(normalize_season)
        else:
            m[col] = pd.to_numeric(m[col], errors="coerce").fillna(0).astype(int)

    # Fill missing text values
    text_cols = [c for c in m.columns if m[c].dtype == "object"]
    m[text_cols] = m[text_cols].fillna("Unknown")

    print("🧹 Cleaned matches data preview:")
    print(m.head(5))
    print(f"✅ Final matches shape: {m.shape}")
    return m

def insert_into_mysql(df: pd.DataFrame):
    """Insert cleaned data into MySQL using SQLAlchemy"""
    engine = get_engine()
    with engine.begin() as conn:
        # Clear existing data if needed
        conn.execute(text("DELETE FROM matches;"))
        # Load data
        df.to_sql("matches", con=conn, if_exists="append", index=False)
    print(f"✅ Inserted {len(df)} records into MySQL table 'matches'.")

if __name__ == "__main__":
    raw_df = load_raw_data()
    matches_df = clean_matches_data(raw_df)
    insert_into_mysql(matches_df)
    print("🎯 Step 2 complete: matches table populated!")
