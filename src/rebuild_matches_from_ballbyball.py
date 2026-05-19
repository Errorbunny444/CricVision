import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from database.db_connections import get_engine

def rebuild_matches_from_ballbyball():
    engine = get_engine()
    print("✅ MySQL Connection Successful!")

    # 1️⃣ Load the big IPL.csv
    df = pd.read_csv("data/IPL.csv", low_memory=False)
    print(f"📥 Loaded {len(df):,} rows and {len(df.columns)} columns")

    # 2️⃣ Extract match-level metadata
    match_level_cols = [
        "match_id", "date", "season", "venue", "city",
        "toss_winner", "toss_decision", "match_won_by",
        "superover_winner", "player_of_match", "event_name"
    ]
    df_meta = df[match_level_cols].drop_duplicates(subset=["match_id"]).reset_index(drop=True)

    # 3️⃣ Find participating teams (team1, team2)
    team_pairs = (
        df.groupby("match_id")[["batting_team", "bowling_team"]]
        .agg(lambda x: list(set(x)))
        .reset_index()
    )

    def get_teams(row):
        teams = set(row["batting_team"] + row["bowling_team"])
        return list(teams)[:2] if len(teams) >= 2 else list(teams) + [None]

    team_pairs[["team1", "team2"]] = team_pairs.apply(get_teams, axis=1, result_type="expand")

    df_final = df_meta.merge(team_pairs[["match_id", "team1", "team2"]], on="match_id", how="left")

    # 4️⃣ Clean and rename columns
    df_final.rename(
        columns={
            "match_won_by": "winner",
            "date": "match_date",
        },
        inplace=True,
    )

    # 5️⃣ Drop any matches with missing essential info
    df_final = df_final.dropna(subset=["team1", "team2", "winner"])

    print(f"✅ Cleaned match-level dataset: {len(df_final):,} matches")

    # 6️⃣ Insert into MySQL
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS matches;"))
    df_final.to_sql("matches", engine, index=False, if_exists="replace")

    print(f"✅ Inserted {len(df_final):,} records into MySQL table 'matches'")
    print("📊 Columns:", df_final.columns.tolist())
    print("🎯 Sample:")
    print(df_final.head(5))

if __name__ == "__main__":
    rebuild_matches_from_ballbyball()
