# src/eda_analysis.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# src/eda_analysis.py
import pandas as pd
import plotly.express as px
from sqlalchemy import text
from database.db_connections import get_engine

# ----------------------------------------------------------
# Connect to MySQL
# ----------------------------------------------------------
engine = get_engine()

# ----------------------------------------------------------
# Helper function
# ----------------------------------------------------------
def run_query(query):
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df

# ----------------------------------------------------------
# EDA Queries on match_summary
# ----------------------------------------------------------

# 🏆 Total matches per season
q1 = """
SELECT season, COUNT(*) AS total_matches
FROM match_summary
GROUP BY season
ORDER BY season;
"""

# 🏟️ Top 10 venues by number of matches
q2 = """
SELECT venue, COUNT(*) AS total_matches
FROM match_summary
WHERE venue IS NOT NULL
GROUP BY venue
ORDER BY total_matches DESC
LIMIT 10;
"""

# 🏙️ Top 10 cities by match count
q3 = """
SELECT city, COUNT(*) AS total_matches
FROM match_summary
WHERE city IS NOT NULL
GROUP BY city
ORDER BY total_matches DESC
LIMIT 10;
"""

# ⏱️ Average number of balls per match (useful stat)
q4 = """
SELECT season, ROUND(AVG(ball_count), 2) AS avg_balls_per_match
FROM match_summary
GROUP BY season
ORDER BY season;
"""

# ⚔️ Matches per season grouped by venue
q5 = """
SELECT season, venue, COUNT(*) AS matches
FROM match_summary
GROUP BY season, venue
ORDER BY season, matches DESC;
"""

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------
def visualize():
    # 1️⃣ Matches per season
    df1 = run_query(q1)
    fig1 = px.line(df1, x="season", y="total_matches", markers=True,
                   title="📈 Total Matches Per Season (2008–2025)")
    fig1.show()

    # 2️⃣ Top 10 Venues
    df2 = run_query(q2)
    fig2 = px.bar(df2, x="venue", y="total_matches", text_auto=True,
                  title="🏟️ Top 10 Venues by Match Count")
    fig2.update_layout(xaxis_tickangle=-45)
    fig2.show()

    # 3️⃣ Top 10 Cities
    df3 = run_query(q3)
    fig3 = px.bar(df3, x="city", y="total_matches", text_auto=True,
                  title="🏙️ Top 10 Cities by Number of Matches")
    fig3.update_layout(xaxis_tickangle=-45)
    fig3.show()

    # 4️⃣ Avg Balls per Match
    df4 = run_query(q4)
    fig4 = px.line(df4, x="season", y="avg_balls_per_match", markers=True,
                   title="⏱️ Average Balls per Match (by Season)")
    fig4.show()

    # 5️⃣ Matches per Season per Venue (heatmap style)
    df5 = run_query(q5)
    fig5 = px.density_heatmap(df5, x="season", y="venue", z="matches", color_continuous_scale="Viridis",
                              title="🔥 Venue Participation per Season")
    fig5.show()

if __name__ == "__main__":
    visualize()

