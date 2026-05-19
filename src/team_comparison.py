# src/team_comparison.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import text
from database.db_connections import get_engine

def team_season_summary(team):
    engine = get_engine()
    q = """
    SELECT season,
           COUNT(DISTINCT CONCAT(season,'-',match_key)) AS matches,
           SUM(CASE WHEN batting_team = :team THEN runs_total ELSE 0 END) AS runs_scored,
           SUM(CASE WHEN bowling_team = :team THEN runs_total ELSE 0 END) AS runs_against
    FROM matches
    GROUP BY season
    ORDER BY season;
    """
    return pd.read_sql(text(q), engine, params={"team": team})

def head_to_head(team_a, team_b):
    engine = get_engine()
    q = """
    SELECT season,
           COUNT(DISTINCT CONCAT(season,'-',match_key)) AS meetings,
           SUM(CASE WHEN winner = :team_a THEN 1 ELSE 0 END) AS wins_a,
           SUM(CASE WHEN winner = :team_b THEN 1 ELSE 0 END) AS wins_b
    FROM match_summary
    WHERE (venue IS NOT NULL)
      AND ( (venue IS NOT NULL) )
      AND ( (season IS NOT NULL) )
      AND ( (1=1) )
    GROUP BY season;
    """
    # We'll compute head-to-head using match_summary by filtering where both teams played — better approach below:
    q2 = """
    SELECT season,
           SUM(CASE WHEN (match_key LIKE concat('%',:a,'%') AND match_key LIKE concat('%',:b,'%') AND winner = :a) THEN 1 ELSE 0 END) AS wins_a,
           SUM(CASE WHEN (match_key LIKE concat('%',:a,'%') AND match_key LIKE concat('%',:b,'%') AND winner = :b) THEN 1 ELSE 0 END) AS wins_b,
           COUNT(DISTINCT CASE WHEN (match_key LIKE concat('%',:a,'%') AND match_key LIKE concat('%',:b,'%')) THEN match_key END) AS meetings
    FROM match_summary
    GROUP BY season
    ORDER BY season;
    """
    return pd.read_sql(text(q2), get_engine(), params={"a": team_a, "b": team_b})
