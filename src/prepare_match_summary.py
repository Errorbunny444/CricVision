import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src/prepare_match_summary.py
# src/prepare_match_summary.py
import pandas as pd
from sqlalchemy import text, inspect
from database.db_connections import get_engine

def create_match_summary():
    engine = get_engine()
    print("✅ MySQL Connection Successful!")

    # Dynamically detect whether column is 'date' or 'match_date'
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("matches")]
    print("📊 Columns in matches table:", columns)

    date_col = "match_date" if "match_date" in columns else "date"

    q = f"""
        SELECT
            match_id, season,
            {date_col} AS match_date,
            city, venue,
            team1, team2, toss_winner, toss_decision,
            winner, player_of_match
        FROM matches;
    """

    print("📥 Running query:\n", q)
    df = pd.read_sql(text(q), engine)
    print(f"✅ Loaded {len(df):,} records from matches")

    df["match_key"] = (
        df["season"].astype(str)
        + "_"
        + df["match_id"].astype(str)
        + "_"
        + df["team1"].astype(str)
        + "_"
        + df["team2"].astype(str)
    )

    df["ball_count"] = 240

    summary = df[
        [
            "season", "match_key", "venue", "city",
            "team1", "team2", "toss_winner", "toss_decision",
            "winner", "ball_count", "player_of_match"
        ]
    ]

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS match_summary;"))
    summary.to_sql("match_summary", engine, index=False, if_exists="replace")

    print(f"✅ match_summary created successfully with {len(summary)} rows")
    print("📊 Columns:", summary.columns.tolist())

if __name__ == "__main__":
    create_match_summary()





