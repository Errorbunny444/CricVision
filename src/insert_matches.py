# src/insert_matches.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from database.db_connections import get_engine

engine = get_engine()

# Load your cleaned IPL CSV (the same one you uploaded)
df = pd.read_csv("data/IPL.csv")

print(f"✅ Loaded {len(df):,} rows from IPL.csv")

# Optional cleanup (depends on your CSV)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# Select key columns (if available)
cols = [c for c in df.columns if c in [
    "match_id", "season", "match_date", "city", "venue",
    "team1", "team2", "toss_winner", "toss_decision",
    "winner", "result", "win_by_runs", "win_by_wickets",
    "player_of_match", "umpire1", "umpire2"
]]
df = df[cols]

print("📊 Columns being inserted:", df.columns.tolist())

# Drop old table and insert fresh data
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS matches;"))

df.to_sql("matches", engine, index=False, if_exists="replace")
print(f"✅ Inserted {len(df):,} rows into MySQL table 'matches'")
