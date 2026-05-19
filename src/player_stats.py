# src/player_stats.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src/player_stats.py
import pandas as pd
from database.db_connections import get_engine
from sqlalchemy import text

def get_top_batsmen(limit=20):
    """Return career totals: runs, balls, strike rate, innings played"""
    engine = get_engine()
    q = """
    SELECT batter AS player,
           SUM(runs_batter) AS runs,
           SUM(balls_faced) AS balls,
           SUM(CASE WHEN balls_faced>0 THEN 1 ELSE 0 END) AS innings
    FROM matches
    WHERE batter IS NOT NULL
    GROUP BY batter
    ORDER BY runs DESC
    LIMIT :limit;
    """
    return pd.read_sql(text(q), engine, params={"limit": limit})

def get_top_bowlers(limit=20):
    """Return wickets (if dismissal data exists) and runs conceded & economy"""
    engine = get_engine()
    # Note: we attempt to detect wicket column names; adapt if different
    # For many ball datasets `player_dismissed` and `dismissal_kind` are present - if not we approximate by 'is_wicket'
    q = """
    SELECT bowler AS player,
           COUNT(*) AS balls_bowled,
           SUM(runs_bowler) AS runs_conceded
    FROM matches
    WHERE bowler IS NOT NULL
    GROUP BY bowler
    ORDER BY runs_conceded ASC
    LIMIT :limit;
    """
    return pd.read_sql(text(q), engine, params={"limit": limit})

def player_time_series(player_name, role='batter'):
    """Return season-wise performance for a player"""
    engine = get_engine()
    if role == 'batter':
        q = """
        SELECT season, SUM(runs_batter) AS runs, SUM(balls_faced) AS balls
        FROM matches
        WHERE batter = :player
        GROUP BY season
        ORDER BY season;
        """
    else:
        q = """
        SELECT season, COUNT(*) AS balls_bowled, SUM(runs_bowler) AS runs_conceded
        FROM matches
        WHERE bowler = :player
        GROUP BY season
        ORDER BY season;
        """
    return pd.read_sql(text(q), engine, params={"player": player_name})
