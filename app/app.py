# app/app.py
import sys
import os
import warnings
from typing import List, Optional, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 
from sqlalchemy import text
from database.db_connections import get_engine
from pyvis.network import Network
import streamlit.components.v1 as components
from sqlalchemy import create_engine
from prophet import Prophet
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# --------------------- Config ---------------------
DB_USER = "root"
DB_PASSWORD = "Dinner!18"
DB_HOST = "localhost"
DB_NAME = "ipl_db"
ENGINE = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CRICVISION — Full IPL Intelligence", layout="wide", page_icon="🏏")
st.title("🏏 CRICVISION — Full IPL Intelligence & Analytics")
st.markdown("A complete spatiotemporal IPL analytics system — from players to predictors.")
st.markdown("---")

# --------------------- DB + Model ---------------------
engine = get_engine()

@st.cache_data(show_spinner=False)
def run_query(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    try:
        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params)
    except Exception as e:
        st.session_state.setdefault("_debug_logs", []).append(str(e))
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return joblib.load("models/match_predictor.pkl")
    except Exception:
        return None

model = load_model()

# --------------------- Helpers ---------------------
def normalize_season_value(season_str):
    try:
        s = str(season_str).strip()
        if "/" in s:
            parts = s.split("/")
            last = parts[-1]
            if len(last) == 2 and last.isdigit():
                return int("20" + last)
            if len(last) == 4 and last.isdigit():
                return int(last)
        if s.isdigit():
            return int(s)
    except Exception:
        pass
    return np.nan

def append_season_filter(base_query: str, season_value: str) -> str:
    q = base_query.strip().rstrip(";")
    if not season_value or season_value == "All Seasons":
        return q + ";"
    if " where " in q.lower():
        return q + f" AND season = '{season_value}';"
    return q + f" WHERE season = '{season_value}';"

@st.cache_data(show_spinner=False)
def table_columns(table_name: str) -> List[str]:
    try:
        df = run_query(f"SHOW COLUMNS FROM {table_name};")
        if not df.empty and "Field" in df.columns:
            return df["Field"].astype(str).tolist()
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False)
def load_match_summary() -> pd.DataFrame:
    df = run_query("SELECT * FROM match_summary;")
    if df.empty:
        return df
    if "season" in df.columns:
        df["season_num"] = df["season"].apply(normalize_season_value)
    return df

match_summary_df = load_match_summary()
matches_cols = table_columns("matches")
deliveries_cols = table_columns("deliveries")

def detect_batsman_runs_col(cols: List[str]) -> Optional[str]:
    for c in cols:
        lc = c.lower()
        if "batsman_runs" in lc or ("bat" in lc and "run" in lc):
            return c
    return None

runs_col = detect_batsman_runs_col(deliveries_cols) or "batsman_runs"
has_over = next((c for c in deliveries_cols if "over" in c.lower()), "over_num")
has_ball = next((c for c in deliveries_cols if "ball" in c.lower()), "ball_num")
has_matchid = next((c for c in deliveries_cols if "match" in c.lower()), "match_id")

def build_team_list():
    teams = set()
    if not match_summary_df.empty:
        for c in ("team1", "team2", "winner"):
            if c in match_summary_df.columns:
                teams.update(match_summary_df[c].dropna().unique())
    if "batting_team" in deliveries_cols:
        tmp = run_query("SELECT DISTINCT batting_team FROM deliveries WHERE batting_team IS NOT NULL LIMIT 1000;")
        if not tmp.empty:
            teams.update(tmp.iloc[:, 0].dropna())
    return sorted([t for t in teams if t and t != "Unknown"])

team_list = build_team_list()

# --------------------- Sidebar ---------------------
st.sidebar.header("Filters")
season_options = ["All Seasons"]
if not match_summary_df.empty and "season" in match_summary_df.columns:
    season_options += sorted(match_summary_df["season"].dropna().unique(), key=lambda s: normalize_season_value(s))
selected_season = st.sidebar.selectbox("Select Season", options=season_options, index=0)

# --------------------- Tabs ---------------------
tabs = st.tabs([
    "🏠 Overview", "🏏 Player Analytics", "🤝 Team Comparison",
    "🏟 Venue & Pitch", "🕸 Partnership Network", "🤖 Match Predictor",
    "🧾 Report Generator", "🧭 Strategy Simulator", "📈 Performance Forecast"
])

# ---------------- Overview ----------------
with tabs[0]:
    st.subheader("🏠 IPL Overview")
    st.markdown("Explore IPL trends, top team performances, and win-rate evolution across all seasons.")

    if match_summary_df.empty:
        st.info("No match_summary data available.")
    else:
        df = match_summary_df.copy()
        if selected_season != "All Seasons":
            df = df[df["season"] == selected_season]

        # KPI cards
        total_matches = len(df)
        total_venues = df["venue"].nunique() if "venue" in df.columns else 0
        teams = len(set(df["team1"]).union(df["team2"])) if {"team1", "team2"}.issubset(df.columns) else len(team_list)

        c1, c2, c3 = st.columns(3)
        c1.metric("🏏 Matches", f"{total_matches}")
        c2.metric("👥 Teams", f"{teams}")
        c3.metric("🏟 Venues", f"{total_venues}")

        st.markdown("---")

        # Color palette
        ipl_colors = [
            "#045093", "#F9CD05", "#E03C31", "#4A148C", "#2E7D32",
            "#D81B60", "#00897B", "#5E35B1", "#F4511E", "#3949AB"
        ]

        # Matches per season / venue
        if selected_season == "All Seasons":
            df_season = (
                df.groupby("season", as_index=False)
                .size()
                .rename(columns={"size": "total_matches"})
            )
            df_season["season_num"] = df_season["season"].apply(normalize_season_value)
            df_season = df_season.sort_values("season_num")

            fig = px.bar(
                df_season,
                x="season_num", y="total_matches",
                text_auto=True,
                color="season_num",
                color_continuous_scale=ipl_colors,
                title="📈 Total Matches per Season"
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Season",
                yaxis_title="Matches",
                showlegend=False,
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white", size=13)
            )
            st.plotly_chart(fig, width="stretch")

        else:
            df_venue = (
                df.groupby("venue", as_index=False)
                .size()
                .rename(columns={"size": "matches"})
                .sort_values("matches", ascending=False)
            )
            fig = px.bar(
                df_venue,
                x="venue", y="matches",
                text_auto=True,
                color="matches",
                color_continuous_scale=ipl_colors,
                title=f"🏟 Matches by Venue — {selected_season}"
            )
            fig.update_layout(
                template="plotly_dark",
                xaxis_tickangle=-45,
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white", size=13)
            )
            st.plotly_chart(fig, width="stretch")

        # ===============================
        # 🏆 TEAM PERFORMANCE SECTION
        # ===============================
        st.markdown("---")
        st.markdown("### 🏆 Team Performance Analytics by Season")

        q_team = """
            WITH team_runs AS (
                SELECT match_id, batting_team, SUM(total_runs) AS total_runs
                FROM deliveries
                GROUP BY match_id, batting_team
            )
            SELECT 
                m.season,
                tr.batting_team AS team,
                ROUND(AVG(tr.total_runs), 2) AS avg_runs,
                COUNT(DISTINCT m.match_id) AS matches
            FROM team_runs tr
            JOIN matches m ON tr.match_id = m.match_id
            GROUP BY m.season, tr.batting_team
            ORDER BY m.season;
        """
        df_team = run_query(q_team)
        if df_team.empty:
            st.info("Not enough data for team analytics.")
        else:
            df_team["season_num"] = df_team["season"].apply(normalize_season_value)

            # Win / loss data (keeping 2020 separate)
            q_wins = """
                SELECT season, winner AS team, COUNT(*) AS wins
                FROM match_summary WHERE winner IS NOT NULL
                GROUP BY season, winner;
            """
            df_wins = run_query(q_wins)
            df_wins["season_num"] = df_wins["season"].apply(normalize_season_value)

            q_matches = """
                SELECT season, team1 AS team FROM match_summary
                UNION ALL
                SELECT season, team2 AS team FROM match_summary;
            """
            df_m = run_query(q_matches)
            df_m["season_num"] = df_m["season"].apply(normalize_season_value)

            team_perf = (
                df_m.groupby(["season_num", "team"])
                .size().reset_index(name="total_matches")
                .merge(df_wins.groupby(["season_num", "team"]).sum().reset_index(),
                       on=["season_num", "team"], how="left")
                .fillna({"wins": 0})
            )
            team_perf["losses"] = team_perf["total_matches"] - team_perf["wins"]
            team_perf["win_pct"] = ((team_perf["wins"] / team_perf["total_matches"]) * 100).round(1)

            # Merge team avg runs + performance
            df_team = df_team.merge(team_perf, on=["season_num", "team"], how="left").fillna(0)

            # IPL Team Colors
            team_colors = {
                "Chennai Super Kings": "#F9CD05",
                "Mumbai Indians": "#045093",
                "Royal Challengers Bangalore": "#DA1818",
                "Kolkata Knight Riders": "#3C1361",
                "Rajasthan Royals": "#EA1A8E",
                "Sunrisers Hyderabad": "#FB643E",
                "Gujarat Titans": "#1C2340",
                "Lucknow Super Giants": "#4CC1A1",
                "Delhi Capitals": "#17449B",
                "Punjab Kings": "#AA4545",
                "Kings XI Punjab": "#AA4545",
                "Delhi Daredevils": "#17449B",
                "Deccan Chargers": "#1E6EA1",
                "Rising Pune Supergiants": "#702963",
                "Pune Warriors": "#6EC1E4",
                "Kochi Tuskers Kerala": "#FF6600",
            }

            # Toggles
            metric_mode = st.radio("Metric", ["Average Runs per Match", "Win Percentage"], horizontal=True)
            view_mode = st.radio("View Mode", ["Grouped", "Stacked"], horizontal=True, index=0)
            barmode = "stack" if view_mode == "Stacked" else "group"

            y_metric = "avg_runs" if metric_mode == "Average Runs per Match" else "win_pct"
            y_label = "Average Runs" if metric_mode == "Average Runs per Match" else "Win Percentage (%)"

            # Custom hover info
            df_team["custom"] = df_team.apply(
                lambda x: f"🏏 <b>{x['team']}</b> — {int(x['season_num'])}<br>"
                          f"📈 Avg Runs: {x['avg_runs']}<br>"
                          f"🎯 Matches: {int(x['total_matches'])}<br>"
                          f"🏆 Wins: {int(x['wins'])} | ❌ Losses: {int(x['losses'])}<br>"
                          f"📊 Win%: {x['win_pct']}%",
                axis=1
            )

            # Build chart
            fig_team = px.bar(
                df_team,
                x="season_num",
                y=y_metric,
                color="team",
                text_auto=".2f",
                color_discrete_map=team_colors,
                title=f"🏆 {metric_mode} — Top Teams by Season"
            )
            fig_team.update_traces(
                textfont_color="white",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=df_team["custom"]
            )
            fig_team.update_layout(
                template="plotly_dark",
                barmode=barmode,
                xaxis_title="Season",
                yaxis_title=y_label,
                font=dict(color="white", size=13),
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                legend_title="Team",
                hoverlabel=dict(bgcolor="#111827", font_size=13, font_color="white"),
                legend=dict(
                    bgcolor="rgba(15,23,32,0.8)",
                    bordercolor="#222",
                    borderwidth=1,
                    font=dict(color="white", size=12)
                ),
                bargap=0.25,
                transition_duration=500
            )
            fig_team.update_yaxes(range=[0, df_team[y_metric].max() * 1.25])
            st.plotly_chart(fig_team, width="stretch")

# ---------------- Player Analytics ----------------
with tabs[1]:
    st.subheader("🏏 Player Analytics")
    st.markdown("Explore batting and bowling insights with ESPN-style dark statcards, trend charts, and a performance heatmap.")

    # Search input
    player_name = st.text_input("Search Player", value="V Kohli")

    # ---------- 🔥 Top 20 Run Scorers ----------
    st.markdown("### 🔥 Top 20 Run Scorers (Overall)")

    q_top = f"""
        SELECT batter, SUM({runs_col}) AS runs
        FROM deliveries
        WHERE batter IS NOT NULL
        GROUP BY batter
        ORDER BY runs DESC
        LIMIT 20;
    """
    df_top = pd.read_sql(text(q_top), engine)
    if not df_top.empty:
        df_top["runs"] = df_top["runs"].astype(int)
        fig_top = px.bar(
            df_top,
            x="batter",
            y="runs",
            text_auto=True,
            color="runs",
            color_continuous_scale=["#045093", "#F9CD05", "#D81B60", "#4A148C"],
            title="🔥 Top 20 Run Scorers in IPL"
        )
        fig_top.update_traces(textfont_color="white")
        fig_top.update_layout(
            template="plotly_dark",
            xaxis_tickangle=-45,
            plot_bgcolor="#0f1720",
            paper_bgcolor="#0f1720",
            font=dict(color="white")
        )
        st.plotly_chart(fig_top, width="stretch")

    st.markdown("---")

    # ---------- 📊 Player Trend and Summaries ----------
    if player_name:
        # Season-wise performance
        q_ts = f"""
            SELECT m.season, SUM(d.{runs_col}) AS runs
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            WHERE d.batter LIKE :p
            GROUP BY m.season
            ORDER BY m.season;
        """
        df_ts = pd.read_sql(text(q_ts), engine, params={"p": f"%{player_name}%"}).fillna(0)
        if not df_ts.empty:
            df_ts["season_num"] = df_ts["season"].apply(normalize_season_value)
            fig = px.line(
                df_ts,
                x="season_num",
                y="runs",
                markers=True,
                title=f"📊 {player_name} — Season-wise Runs",
                color_discrete_sequence=["#FFB612"]
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Season",
                yaxis_title="Runs",
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white")
            )
            st.plotly_chart(fig, width="stretch")

        # ---------- ESPN-style Statcards ----------
        def statcard_html(cards: Dict[str, str], title: str) -> str:
            card_html = "<div style='display:flex;gap:12px;flex-wrap:wrap;margin-top:8px;'>"
            for label, value in cards.items():
                card_html += f"""
                <div style="background:#0f1720;padding:14px;border-radius:12px;min-width:160px;box-shadow:0 6px 18px rgba(0,0,0,0.6);">
                    <div style="color:#9ca3af;font-size:12px;margin-bottom:6px;">{label}</div>
                    <div style="color:white;font-weight:700;font-size:20px;">{value}</div>
                </div>"""
            card_html += "</div>"
            return f"<div style='margin-top:10px'><div style='color:#e5e7eb;font-weight:700;margin-bottom:6px'>{title}</div>{card_html}</div>"

        # 🏏 Batting Summary
        q_bat = f"""
            WITH innings_data AS (
                SELECT match_id, inning,
                       SUM({runs_col}) AS runs, COUNT(*) AS balls
                FROM deliveries
                WHERE batter LIKE :p
                GROUP BY match_id, inning
            )
            SELECT COUNT(DISTINCT match_id) AS matches, COUNT(*) AS innings,
                   SUM(runs) AS total_runs, MAX(runs) AS high_score,
                   SUM(CASE WHEN runs>=50 AND runs<100 THEN 1 ELSE 0 END) AS fifties,
                   SUM(CASE WHEN runs>=100 THEN 1 ELSE 0 END) AS hundreds,
                   ROUND(AVG(runs),2) AS average,
                   ROUND((SUM(runs)/NULLIF(SUM(balls),0))*100,2) AS strike_rate
            FROM innings_data;
        """
        df_bat = pd.read_sql(text(q_bat), engine, params={"p": f"%{player_name}%"}).fillna(0)
        b = df_bat.iloc[0].to_dict() if not df_bat.empty else {}
        batting_cards = {
            "Matches": f"{int(b.get('matches', 0)):,}",
            "Innings": f"{int(b.get('innings', 0)):,}",
            "Runs": f"{int(b.get('total_runs', 0)):,}",
            "Highest": f"{int(b.get('high_score', 0)):,}",
            "50s": f"{int(b.get('fifties', 0)):,}",
            "100s": f"{int(b.get('hundreds', 0)):,}",
            "Avg": f"{float(b.get('average', 0)):.2f}",
            "SR": f"{float(b.get('strike_rate', 0)):.2f}"
        }
        st.markdown(statcard_html(batting_cards, "🏏 Batting Summary (Career)"), unsafe_allow_html=True)

        # 🎯 Bowling Summary
        q_bowl = """
            WITH bowl_data AS (
                SELECT match_id, inning,
                       SUM(runs_bowler) AS runs_conceded,
                       SUM(is_wicket) AS wickets,
                       COUNT(*) AS balls
                FROM deliveries
                WHERE bowler LIKE :p
                GROUP BY match_id, inning
            )
            SELECT COUNT(DISTINCT match_id) AS matches, COUNT(*) AS innings,
                   SUM(wickets) AS total_wickets, SUM(runs_conceded) AS total_runs,
                   ROUND(SUM(runs_conceded)/NULLIF(SUM(wickets),0),2) AS bowling_average,
                   ROUND((SUM(runs_conceded)/NULLIF(SUM(balls),0))*6,2) AS economy
            FROM bowl_data;
        """
        df_bowl = pd.read_sql(text(q_bowl), engine, params={"p": f"%{player_name}%"}).fillna(0)
        b = df_bowl.iloc[0].to_dict() if not df_bowl.empty else {}
        bowling_cards = {
            "Matches": f"{int(b.get('matches', 0)):,}",
            "Innings": f"{int(b.get('innings', 0)):,}",
            "Wickets": f"{int(b.get('total_wickets', 0)):,}",
            "Runs Conceded": f"{int(b.get('total_runs', 0)):,}",
            "Avg": f"{float(b.get('bowling_average', 0)):.2f}",
            "Econ": f"{float(b.get('economy', 0)):.2f}"
        }
        st.markdown(statcard_html(bowling_cards, "🎯 Bowling Summary (Career)"), unsafe_allow_html=True)

        # ---------- 🔥 Interactive Performance Heatmap ----------
        st.markdown("---")
        st.markdown("### 🔥 Performance Heatmap — Player Consistency by Season")

        # Slider for filtering top players
        top_n = st.slider("Number of players to display", 10, 50, 25)

        q_heat = f"""
            SELECT m.season, d.batter, SUM(d.{runs_col}) AS runs
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            WHERE d.batter IS NOT NULL
            GROUP BY m.season, d.batter
            HAVING SUM(d.{runs_col}) > 200
            ORDER BY m.season;
        """
        df_heat = pd.read_sql(text(q_heat), engine)
        if not df_heat.empty:
            df_heat["season_num"] = df_heat["season"].apply(normalize_season_value)
            df_heat = (
                df_heat.groupby(["batter", "season_num"], as_index=False)["runs"]
                .sum()
                .sort_values(["batter", "season_num"])
            )

            # Select top consistent players
            top_batters = df_heat.groupby("batter")["runs"].sum().nlargest(top_n).index
            pivot = df_heat[df_heat["batter"].isin(top_batters)]
            pivot = pivot.pivot(index="batter", columns="season_num", values="runs").fillna(0)

            fig_heat = px.imshow(
                pivot,
                color_continuous_scale=["#0f1720", "#045093", "#FFB612", "#E03C31"],
                title="🔥 Player Performance Heatmap — Season Consistency",
                aspect="auto"
            )
            fig_heat.update_layout(
                template="plotly_dark",
                xaxis_title="Season",
                yaxis_title="Player",
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white"),
                coloraxis_colorbar=dict(title="Runs", tickfont=dict(color="white"))
            )
            fig_heat.update_xaxes(type="category", showgrid=False)
            fig_heat.update_yaxes(showgrid=False)

            st.plotly_chart(fig_heat, width="stretch")
        else:
            st.info("Not enough data for performance heatmap.")

# ---------------- Team Comparison ----------------
with tabs[2]:
    st.subheader("🤝 Team Comparison & Head-to-Head")
    if not team_list:
        st.info("No teams found in dataset.")
    else:
        team_a = st.selectbox("Team A", options=team_list)
        team_b = st.selectbox("Team B", options=[t for t in team_list if t != team_a])

        if match_summary_df.empty:
            st.info("No match_summary data to compute head-to-head.")
        else:
            mask = (
                ((match_summary_df.get("team1") == team_a) & (match_summary_df.get("team2") == team_b))
                | ((match_summary_df.get("team1") == team_b) & (match_summary_df.get("team2") == team_a))
            )
            df_h2h = match_summary_df[mask]
            if df_h2h.empty:
                st.info("No head-to-head matches found.")
            else:
                win_counts = df_h2h["winner"].value_counts().reset_index()
                win_counts.columns = ["team", "wins"]
                fig_h2h = px.bar(win_counts, x="team", y="wins", text_auto=True, title=f"Head-to-Head: {team_a} vs {team_b}", color="wins")
                fig_h2h.update_layout(template="plotly_dark")
                st.plotly_chart(fig_h2h, width="stretch")

                df_h2h_by_season = df_h2h.groupby("season", as_index=False).size().rename(columns={"size": "meetings"})
                if not df_h2h_by_season.empty:
                    df_h2h_by_season["season_num"] = df_h2h_by_season["season"].apply(normalize_season_value)
                    df_h2h_by_season = df_h2h_by_season.sort_values("season_num")
                    fig_meet = px.bar(df_h2h_by_season, x="season_num", y="meetings", title="Meetings per Season")
                    fig_meet.update_layout(template="plotly_dark", xaxis_title="Season")
                    st.plotly_chart(fig_meet, width="stretch")

                total_wins = df_h2h["winner"].value_counts().reset_index()
                total_wins.columns = ["Team", "Wins"]
                if not total_wins.empty:
                    fig_pie = px.pie(total_wins, names="Team", values="Wins", hole=0.35, title="Win Share")
                    fig_pie.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_pie, width="stretch")

# ---------------- Venue & Pitch ----------------
with tabs[3]:
    st.subheader("🏟 Venue & Pitch Insights")
    st.markdown("Venue counts and batting-first vs chasing outcomes (best-effort).")

    q_venue = "SELECT venue, COUNT(*) AS matches FROM match_summary WHERE venue IS NOT NULL GROUP BY venue ORDER BY matches DESC LIMIT 20"
    if selected_season != "All Seasons":
        q_venue = append_season_filter(q_venue, selected_season)
    else:
        q_venue = q_venue + ";"
    df_v = run_query(q_venue)

    if not df_v.empty:
        fig_v = px.bar(df_v, x="venue", y="matches", text_auto=True, title="🏟 Top Venues by Matches Played", color="matches")
        fig_v.update_layout(template="plotly_dark", xaxis_tickangle=-45)
        st.plotly_chart(fig_v, width="stretch")
    else:
        st.info("No venue data available for selected season.")

    st.markdown("### ⚖️ Venue: Batting-first vs Chasing (best-effort)")
    if not match_summary_df.empty and {"winner", "toss_decision", "toss_winner"}.issubset(match_summary_df.columns):
        df_local = match_summary_df.copy()
        if selected_season != "All Seasons":
            df_local = df_local[df_local["season"] == selected_season]
        if df_local.empty:
            st.info("No data for selected season.")
        else:
            df_local["batting_first_winner_flag"] = ((df_local["toss_decision"].str.lower() == "bat") & (df_local["toss_winner"] == df_local["winner"])).astype(int)
            df_local["chase_winner_flag"] = ((df_local["toss_decision"].str.lower() == "field") & (df_local["toss_winner"] == df_local["winner"])).astype(int)
            venue_stats = df_local.groupby("venue", as_index=False).agg(
                bat_first_wins=("batting_first_winner_flag", "sum"),
                chase_wins=("chase_winner_flag", "sum"),
                total_matches=("winner", "count"),
            ).sort_values("total_matches", ascending=False).head(20)
            if not venue_stats.empty:
                df_melted = venue_stats.melt(id_vars=["venue", "total_matches"], var_name="result_type", value_name="count")
                fig_vpitch = px.bar(df_melted, x="venue", y="count", color="result_type", title="Heuristic: Bat-first vs Chase wins by venue")
                fig_vpitch.update_layout(template="plotly_dark", xaxis_tickangle=-45)
                st.plotly_chart(fig_vpitch, width="stretch")
            else:
                st.info("Not enough data to infer pitch behaviour.")
    else:
        st.info("Pitch outcome columns not available in match_summary.")

# ---------------- Partnership Network ----------------
# ---------------- Improved Team Synergy (NetworkX + PyVis) ----------------
with tabs[4]:
    st.subheader("🕸 Team Synergy & Player Influence (NetworkX + PyVis)")
    st.markdown(
        "Interactive partnership / synergy graph built from ball-level deliveries. "
        "Use Team/Season filters to narrow down. PyVis graph is interactive (drag nodes)."
    )

    # Defensive column detection (use the variables you already have)
    batter_col = next((c for c in deliveries_cols if c.lower() in ("batter", "batsman", "batter_name")), "batter")
    non_striker_col = next((c for c in deliveries_cols if c.lower() in ("non_striker", "nonstriker", "non_striker_name")), "non_striker")
    run_col = runs_col or next((c for c in deliveries_cols if "run" in c.lower()), "batsman_runs")
    match_col = next((c for c in deliveries_cols if c.lower() in ("match_id", "matchid")), "match_id")
    batting_team_col = next((c for c in deliveries_cols if c.lower() in ("batting_team", "battingteam")), "batting_team")
    season_available = "season" in match_summary_df.columns if match_summary_df is not None else False

    if not (batter_col in deliveries_cols and non_striker_col in deliveries_cols):
        st.error("deliveries table must contain batter and non_striker columns to show synergy graph.")
    else:
        # UI controls (unique keys to avoid duplicate-element issues)
        with st.expander("Graph options", expanded=True):
            max_rows = st.number_input("Max deliveries to fetch (safety)", min_value=5000, max_value=1000000, value=80000, step=5000, key="team_synergy_maxrows")
            team_options = ["All Teams"]
            # build team options from match_summary or deliveries fallback
            if not match_summary_df.empty and {"team1", "team2"}.issubset(match_summary_df.columns):
                teams_from_matches = sorted(set(match_summary_df["team1"].dropna().tolist() + match_summary_df["team2"].dropna().tolist()))
                team_options = ["All Teams"] + teams_from_matches
            else:
                if batting_team_col in deliveries_cols:
                    tmp = run_query(f"SELECT DISTINCT {batting_team_col} FROM deliveries WHERE {batting_team_col} IS NOT NULL LIMIT 500;")
                    if not tmp.empty:
                        team_options = ["All Teams"] + sorted(tmp.iloc[:,0].dropna().astype(str).tolist())

            selected_team = st.selectbox("Filter by Team", options=team_options, index=0, key="team_synergy_team")
            season_filter = "All Seasons"
            if season_available:
                seasons = ["All Seasons"] + sorted(match_summary_df["season"].dropna().unique().tolist(), key=lambda s: normalize_season_value(s))
                season_filter = st.selectbox("Season", options=seasons, index=0, key="team_synergy_season")
            min_edge_weight = st.slider("Minimum partnership weight to show (edges)", 1, 20, 2, key="team_synergy_minw")
            node_limit = st.number_input("Max nodes to display (reduces clutter)", min_value=20, max_value=1000, value=200, step=10, key="team_synergy_nodelimit")

        # Build SQL with filters (defensive)
        where_clauses = []
        params = {}
        if selected_team and selected_team != "All Teams":
            where_clauses.append(f"{batting_team_col} = :team")
            params["team"] = selected_team
        if season_available and season_filter and season_filter != "All Seasons":
            # join to matches to filter by season
            season_join = True
            params["season"] = season_filter
        else:
            season_join = False

        # Compose query
        if season_join:
            sql = f"""
                SELECT d.{match_col} as match_id, d.{batting_team_col} as batting_team,
                       d.{batter_col} as batter, d.{non_striker_col} as non_striker, d.{run_col} as runs
                FROM deliveries d
                JOIN matches m ON d.{match_col} = m.match_id
                WHERE m.season = :season
            """
            if selected_team and selected_team != "All Teams":
                sql += " AND d.{batting_team_col} = :team".format(batting_team_col=batting_team_col)
        else:
            sql = f"""
                SELECT {match_col} as match_id, {batting_team_col} as batting_team,
                       {batter_col} as batter, {non_striker_col} as non_striker, {run_col} as runs
                FROM deliveries
                WHERE {batter_col} IS NOT NULL AND {non_striker_col} IS NOT NULL
            """
            if selected_team and selected_team != "All Teams":
                sql += f" AND {batting_team_col} = :team"

        sql += f" LIMIT {int(max_rows)};"

        # Fetch
        df_pairs = run_query(sql, params=params)
        if df_pairs.empty:
            st.info("No partnership rows found for the chosen filters.")
        else:
            # normalize column names in pandas
            df_pairs = df_pairs.rename(columns={batter_col: "batter", non_striker_col: "non_striker", run_col: "runs", batting_team_col: "batting_team", match_col: "match_id"})
            # sanitize types
            df_pairs["batter"] = df_pairs["batter"].astype(str).str.strip()
            df_pairs["non_striker"] = df_pairs["non_striker"].astype(str).str.strip()
            df_pairs["runs"] = pd.to_numeric(df_pairs["runs"], errors="coerce").fillna(0).astype(int)

            # Group by pair and accumulate weight (partnership runs) and appearances
            # Normalize ordering so (A,B) and (B,A) map to same key
            def ordered_pair(a, b):
                return (a, b) if a <= b else (b, a)

            grp = {}
            appearances = {}
            for _, row in df_pairs.iterrows():
                a = row["batter"]
                b = row["non_striker"]
                if a == "" or b == "" or pd.isna(a) or pd.isna(b):
                    continue
                key = ordered_pair(a, b)
                grp.setdefault(key, 0)
                grp[key] += int(row["runs"])
                appearances.setdefault(key, 0)
                appearances[key] += 1

            # Construct DataFrame of edges
            edges = []
            for (a, b), runs_sum in grp.items():
                edges.append({"a": a, "b": b, "runs_together": runs_sum, "appearances": appearances.get((a,b), appearances.get((b,a), 0))})
            df_edges = pd.DataFrame(edges)
            if df_edges.empty:
                st.info("No valid partnerships after aggregation.")
            else:
                # Filter by min_edge_weight
                df_edges = df_edges[df_edges["appearances"] >= min_edge_weight].sort_values("runs_together", ascending=False)
                if df_edges.empty:
                    st.info("No edges passed the minimum weight filter. Reduce the threshold.")
                else:
                    # Build NetworkX graph
                    G = nx.Graph()
                    for _, r in df_edges.iterrows():
                        a, b = r["a"], r["b"]
                        w = int(r["runs_together"])
                        ap = int(r["appearances"])
                        if G.has_edge(a, b):
                            G[a][b]["weight"] += w
                            G[a][b]["appearances"] += ap
                        else:
                            G.add_edge(a, b, weight=w, appearances=ap)

                    # Optionally trim nodes to top-N by degree or weight to avoid huge graph
                    if G.number_of_nodes() > node_limit:
                        # keep top nodes by total edge weight
                        node_weights = {n: sum(d["weight"] for _,_,d in G.edges(n,data=True)) for n in G.nodes()}
                        top_nodes = set(sorted(node_weights, key=node_weights.get, reverse=True)[:node_limit])
                        G = G.subgraph(top_nodes).copy()

                    st.success(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges (after filters).")

                    # -------- Enhanced PyVis Graph Styling for Visibility --------
                    net = Network(height="720px", width="100%", bgcolor="#0f1720", font_color="white", notebook=False)
                    net.force_atlas_2based(gravity=-30, central_gravity=0.02, spring_length=120, spring_strength=0.08)
                    # Define a color scale (brighter for stronger players)
                    import matplotlib.cm as cm
                    import matplotlib.colors as mcolors
                    
                    cmap = cm.get_cmap("plasma")
                    max_weight = max(sum(d["weight"] for _,_,d in G.edges(n, data=True)) for n in G.nodes())
                    
                    # Add nodes — use brighter color for higher run contribution
                    for n in G.nodes():
                        total_w = sum(d["weight"] for _,_,d in G.edges(n, data=True))
                        deg = G.degree(n)
                        normalized = total_w / max_weight if max_weight else 0.2
                        color_hex = mcolors.to_hex(cmap(normalized))
                        title = f"<b>{n}</b><br>Total Runs: {total_w}<br>Partnerships: {deg}"
                        net.add_node(
                            n,
                            label=n,
                            title=title,
                            value=max(2, math.log1p(total_w)),
                            color=color_hex,
                            borderWidth=2,
                            borderWidthSelected=4
                            )
                    # Add edges — semi-transparent bright yellowish lines
                    for u, v, d in G.edges(data=True):
                        w = d.get("weight", 1)
                        ap = d.get("appearances", 1)
                        width = max(1, min(20, math.sqrt(w)))
                        title = f"🏏 {u} & {v}<br>Runs together: {w}<br>Appearances: {ap}"
                        net.add_edge(
                            u, v,
                            value=w,
                            title=title,
                            width=width,
                            color="rgba(255,215,0,0.4)"  # bright visible edge (goldish)
                        )

                    # Enable physics & improved visibility
                    net.toggle_physics(True)
                    net.show_buttons(filter_=['physics'])

                    # Save and embed
                    tmp_html = "team_synergy_network.html"
                    net.save_graph(tmp_html)
                    with open(tmp_html, "r", encoding="utf-8") as f:
                        html = f.read()
                    components.html(html, height=720, scrolling=True)

                    # Show top partnerships table
                    st.markdown("#### 🔍 Top Partnerships (by runs together)")
                    st.dataframe(df_edges[["a", "b", "runs_together", "appearances"]].head(25).rename(columns={"a":"Player A","b":"Player B","runs_together":"Runs together","appearances":"Appearances"}), use_container_width=True)


# ---------------- Match Predictor ----------------
with tabs[5]:
    st.subheader("🤖 Match Outcome Predictor")
    st.markdown("Predict match winner (baseline model). For best results train improved model.")

    if model is None:
        st.warning("No trained model found. Please run `src/model_training.py` to create models/match_predictor.pkl")
    else:
        c1, c2 = st.columns(2)
        with c1:
            team1 = st.selectbox("Team 1", options=team_list)
            team2 = st.selectbox("Team 2", options=[t for t in team_list if t != team1])
            toss_winner = st.selectbox("Toss Winner", options=[team1, team2])
        with c2:
            venue_options = match_summary_df["venue"].dropna().unique().tolist() if not match_summary_df.empty and "venue" in match_summary_df.columns else ["Unknown"]
            venue = st.selectbox("Venue", options=sorted(venue_options))
            toss_decision = st.radio("Toss Decision", ["bat", "field"])
            season = st.selectbox("Season", options=sorted(match_summary_df["season"].dropna().unique().tolist()) if not match_summary_df.empty else ["2025"])

        if st.button("🔮 Predict Winner"):
            if team1 == team2:
                st.error("Select two different teams.")
            else:
                test_df = pd.DataFrame([{
                    "team1": team1, "team2": team2, "venue": venue,
                    "toss_winner": toss_winner, "toss_decision": toss_decision, "season": season
                }])
                try:
                    pred = model.predict(test_df)[0]
                    proba = model.predict_proba(test_df)[0]
                    st.success(f"🏆 Predicted Winner: **{pred}**")
                    fig_prob = px.bar(x=model.classes_, y=proba * 100, title="📊 Win Probability (%)", labels={"x":"Team","y":"Probability (%)"}, color=proba * 100)
                    fig_prob.update_layout(template="plotly_dark", xaxis_tickangle=-45)
                    st.plotly_chart(fig_prob, width="stretch")

                    confidence = float(np.max(proba) * 100)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence,
                        title={'text': "Prediction Confidence (%)"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "limegreen"},
                               'steps': [{'range': [0, 50], 'color': "lightcoral"},
                                         {'range': [50, 75], 'color': "gold"},
                                         {'range': [75, 100], 'color': "lightgreen"}]}
                    ))
                    fig_gauge.update_layout(template="plotly_dark")
                    st.plotly_chart(fig_gauge, width="stretch")

                    # toss+venue historic insight (best-effort)
                    if not match_summary_df.empty and {"venue", "toss_decision", "winner"}.issubset(match_summary_df.columns):
                        q_toss = """
                            SELECT toss_decision, COUNT(*) AS matches,
                                   SUM(CASE WHEN toss_winner = winner THEN 1 ELSE 0 END) AS toss_winner_wins
                            FROM match_summary
                            WHERE venue = :venue
                            GROUP BY toss_decision;
                        """
                        df_toss = pd.read_sql(text(q_toss), engine, params={"venue": venue})
                        if not df_toss.empty:
                            st.markdown("#### ⚙️ Toss + Venue Insights (historic)")
                            st.dataframe(df_toss, width="stretch")
                except Exception as e:
                    st.error("Prediction failed: " + str(e))

# ---------------- PDF Report Generator (No wkhtmltopdf needed) ----------------
with tabs[6]:
    st.subheader("🧾 Report Generator — Export Analytics as PDF")
    st.markdown("Generate and download a professional PDF summary of your player or season analytics.")

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    report_type = st.radio("Choose report type:", ["Player Report", "Overview Report"], horizontal=True)

    def generate_pdf_report_reportlab(title, content_lines, filename="cricvision_report.pdf"):
        pdf = SimpleDocTemplate(filename, pagesize=A4, title=title)
        story = []
        styles = getSampleStyleSheet()
        story.append(Paragraph(f"<b><font size=16 color='#FFB612'>{title}</font></b>", styles["Title"]))
        story.append(Spacer(1, 0.2 * inch))
        for line in content_lines:
            story.append(Paragraph(line, styles["Normal"]))
            story.append(Spacer(1, 0.1 * inch))
        pdf.build(story)
        return filename

    if report_type == "Player Report":
        player_name = st.text_input("Enter Player Name", value="V Kohli")
        if st.button("📄 Generate Player Report"):
            st.info("Generating player analytics report… please wait.")
            try:
                q_bat = f"""
                    WITH innings_data AS (
                        SELECT match_id, inning, SUM({runs_col}) AS runs, COUNT(*) AS balls
                        FROM deliveries
                        WHERE batter LIKE :p
                        GROUP BY match_id, inning
                    )
                    SELECT COUNT(DISTINCT match_id) AS matches, COUNT(*) AS innings,
                           SUM(runs) AS total_runs, MAX(runs) AS high_score,
                           SUM(CASE WHEN runs>=50 AND runs<100 THEN 1 ELSE 0 END) AS fifties,
                           SUM(CASE WHEN runs>=100 THEN 1 ELSE 0 END) AS hundreds,
                           ROUND(AVG(runs),2) AS average,
                           ROUND((SUM(runs)/NULLIF(SUM(balls),0))*100,2) AS strike_rate
                    FROM innings_data;
                """
                df_bat = pd.read_sql(text(q_bat), engine, params={"p": f"%{player_name}%"}).fillna(0)
                stats = df_bat.iloc[0].to_dict() if not df_bat.empty else {}

                # Create readable content
                content = [
                    f"<font color='white'><b>Player:</b> {player_name}</font>",
                    f"Matches: {int(stats.get('matches',0))}",
                    f"Innings: {int(stats.get('innings',0))}",
                    f"Total Runs: {int(stats.get('total_runs',0))}",
                    f"Highest Score: {int(stats.get('high_score',0))}",
                    f"50s: {int(stats.get('fifties',0))}",
                    f"100s: {int(stats.get('hundreds',0))}",
                    f"Average: {float(stats.get('average',0)):.2f}",
                    f"Strike Rate: {float(stats.get('strike_rate',0)):.2f}",
                    "<br/><br/><i>Generated using CRICVISION Intelligent IPL Analytics</i>"
                ]

                pdf_filename = f"{player_name.replace(' ', '_')}_report.pdf"
                generate_pdf_report_reportlab(f"CRICVISION Player Report: {player_name}", content, pdf_filename)
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="⬇️ Download Player Report PDF",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                st.success(f"Player report for {player_name} generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")

    else:
        if st.button("📊 Generate Overview Report"):
            st.info("Generating overview report… please wait.")
            try:
                total_matches = len(match_summary_df)
                total_teams = len(set(match_summary_df["team1"]).union(match_summary_df["team2"]))
                total_venues = match_summary_df["venue"].nunique()
                top_winners = match_summary_df["winner"].value_counts().head(5)

                data = [["Team", "Wins"]] + [[t, str(w)] for t, w in zip(top_winners.index, top_winners.values)]
                table = Table(data, colWidths=[3 * inch, 1 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('BOX', (0, 0), (-1, -1), 1, colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
                ]))

                pdf_filename = "ipl_overview_report.pdf"
                pdf = SimpleDocTemplate(pdf_filename, pagesize=A4, title="IPL Overview Report")
                story = []
                styles = getSampleStyleSheet()
                story.append(Paragraph("<b><font size=16 color='#FFB612'>CRICVISION IPL Overview Report</font></b>", styles["Title"]))
                story.append(Spacer(1, 0.2 * inch))
                story.append(Paragraph(f"<b>Total Matches:</b> {total_matches}", styles["Normal"]))
                story.append(Paragraph(f"<b>Total Teams:</b> {total_teams}", styles["Normal"]))
                story.append(Paragraph(f"<b>Total Venues:</b> {total_venues}", styles["Normal"]))
                story.append(Spacer(1, 0.3 * inch))
                story.append(Paragraph("<b>Top 5 Winning Teams:</b>", styles["Heading2"]))
                story.append(table)
                story.append(Spacer(1, 0.3 * inch))
                story.append(Paragraph("<i>Auto-generated report © CRICVISION</i>", styles["Normal"]))
                pdf.build(story)

                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(
                        label="⬇️ Download Overview Report PDF",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )
                st.success("Overview report generated successfully!")
            except Exception as e:
                st.error(f"Error generating report: {e}")

with tabs[-1]:
    st.subheader("📈 Performance Forecast — Calibrated Realistic Model")
    st.markdown("Forecast future IPL performance with statistically adjusted Prophet model for realistic projections.")

    player_name = st.text_input("🔍 Player for Forecast", value="V Kohli")
    future_seasons = st.number_input("Forecast next N seasons", value=2, min_value=1, max_value=5, step=1)
    changepoint = st.slider("Model sensitivity (changepoint_prior_scale)", 0.001, 1.0, 0.05, 0.005)

    if player_name:
        q_perf = f"""
            SELECT
                m.season,
                COUNT(DISTINCT d.match_id) AS matches_played,
                SUM(d.{runs_col}) AS total_runs
            FROM deliveries d
            JOIN matches m ON d.match_id = m.match_id
            WHERE d.batter LIKE :p
            GROUP BY m.season
            ORDER BY m.season;
        """
        df_perf = pd.read_sql(text(q_perf), engine, params={"p": f"%{player_name}%"})

        if df_perf.empty:
            st.warning("No historical data found for that player.")
        else:
            df_perf["matches_played"] = pd.to_numeric(df_perf["matches_played"], errors="coerce").fillna(0)
            df_perf["total_runs"] = pd.to_numeric(df_perf["total_runs"], errors="coerce").fillna(0)
            df_perf = df_perf[df_perf["matches_played"] > 0]
            df_perf["runs_per_match"] = df_perf["total_runs"] / df_perf["matches_played"]

            def season_to_date(s):
                return datetime(int(str(s)[:4]), 4, 1)
            df_perf["ds"] = df_perf["season"].apply(season_to_date)
            df_perf = df_perf.sort_values("ds")

            # Prophet with conservative growth
            model = Prophet(
                yearly_seasonality=True,
                changepoint_prior_scale=float(changepoint),
                seasonality_mode="additive",
                growth="flat"
            )
            model.fit(df_perf.rename(columns={"runs_per_match": "y", "ds": "ds"})[["ds", "y"]])

            last_season = df_perf["ds"].max()
            last_year = last_season.year
            median_matches = int(df_perf["matches_played"].median())
            expected_matches = st.number_input("Expected matches per future season", value=median_matches if median_matches > 0 else 14, min_value=1, max_value=30)

            # Create realistic future timeline (start from next season)
            future_dates = pd.date_range(
                start=datetime(last_year + 1, 4, 1),
                periods=future_seasons,
                freq="YS"
            )
            future_df = pd.DataFrame({"ds": future_dates})
            forecast = model.predict(future_df)

            # Prophet predictions (runs per match)
            forecast["yhat"] = forecast["yhat"].clip(lower=0)
            forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
            forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

            # 🧮 Statistical Recalibration
            recent_mean = df_perf["runs_per_match"].tail(5).mean()
            recent_std = df_perf["runs_per_match"].tail(5).std()

            # smooth Prophet trend toward recent historical stats
            forecast["adjusted_yhat"] = (
                0.6 * forecast["yhat"] + 
                0.4 * np.clip(np.random.normal(recent_mean, recent_std * 0.5, len(forecast)), 0, 100)
            )

            # Realistic scaling for total runs
            forecast["pred_runs_total"] = forecast["adjusted_yhat"] * expected_matches
            forecast["pred_runs_total"] = forecast["pred_runs_total"].clip(lower=200, upper=900)

            # merge historical
            hist_x, hist_y = df_perf["ds"], df_perf["runs_per_match"]

            # plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_x, y=hist_y,
                mode="lines+markers",
                name="Actual Runs/Match",
                line=dict(color="#FFB612", width=3)
            ))
            fig.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["adjusted_yhat"],
                mode="lines+markers",
                name="Predicted Runs/Match",
                line=dict(color="#00E396", width=3, dash="dash")
            ))
            fig.update_layout(
                title=f"📊 {player_name} — Forecasted Runs/Match Trend (Calibrated)",
                template="plotly_dark",
                xaxis_title="Season",
                yaxis_title="Runs per Match",
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white"),
                legend=dict(bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig, use_container_width=True)

            # 🎯 Forecast summary
            forecast["Season"] = forecast["ds"].dt.year
            summary = forecast[["Season", "pred_runs_total"]].copy()
            summary.columns = ["Season", "Predicted Runs"]
            summary["Predicted Runs"] = summary["Predicted Runs"].round(0).astype(int)

            st.markdown("### ✨ Forecast Summary (Realistic Season Totals)")
            st.dataframe(summary, use_container_width=True)

with tabs[-2]:
    st.subheader("🧭 Strategy Simulator")
    st.markdown("Simulate *“What if Team A bats first?”* scenarios using CRICAI’s intelligent ML model.")

    if match_summary_df.empty:
        st.warning("Match summary data not found.")
    else:
        df = match_summary_df.copy()

        # ✅ Normalize season (convert '2007/08' → 2008)
        if "season" in df.columns:
            df["season_num"] = df["season"].apply(normalize_season_value)
        else:
            st.error("Season column missing in match_summary.")
            st.stop()

        # ✅ Aggregate team totals if not available
        if not {"team1_runs", "team2_runs"}.issubset(df.columns):
            st.info("Aggregating team total runs from deliveries table...")
            q_team_scores = """
                SELECT 
                    m.match_id,
                    m.season,
                    d.batting_team AS team,
                    SUM(d.total_runs) AS total_runs
                FROM deliveries d
                JOIN matches m ON d.match_id = m.match_id
                GROUP BY m.match_id, m.season, d.batting_team;
            """
            team_scores = pd.read_sql(text(q_team_scores), engine)
            df_team1 = team_scores.groupby(["match_id", "season"]).nth(0).reset_index()
            df_team2 = team_scores.groupby(["match_id", "season"]).nth(1).reset_index()
            df["team1_runs"] = df_team1["total_runs"].fillna(0)
            df["team2_runs"] = df_team2["total_runs"].fillna(0)

        # Drop incomplete records
        df = df.dropna(subset=["team1", "team2", "venue", "season_num"])

        # Encode teams/venues as numeric
        for col in ["team1", "team2", "venue"]:
            df[col] = df[col].astype(str)
        encoders = {col: LabelEncoder().fit(df[col]) for col in ["team1", "team2", "venue"]}
        for col, le in encoders.items():
            df[col] = le.transform(df[col])

        df["season_num"] = pd.to_numeric(df["season_num"], errors="coerce").fillna(0).astype(int)

        # 🎯 Train regression model
        X = df[["team1", "team2", "venue", "season_num"]]
        y = df["team1_runs"].astype(float)
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)

        # 🎮 User input UI
        st.markdown("### 🏏 Select Simulation Parameters")

        team_a = st.selectbox("Team A", encoders["team1"].classes_, index=0, key="sim_team_a")
        team_b = st.selectbox("Team B", encoders["team2"].classes_, index=1, key="sim_team_b")
        venue = st.selectbox("Venue", encoders["venue"].classes_, key="sim_venue")
        toss_winner = st.selectbox("Toss Winner", [team_a, team_b], key="sim_toss")
        decision = st.radio("Decision", ["Bat", "Field"], key="sim_decision")
        season_input = st.number_input("Season (e.g., 2026)", value=2026, min_value=2008, max_value=2030, key="sim_season")

        if st.button("🎮 Simulate Match", key="sim_run"):
            team_a_encoded = encoders["team1"].transform([team_a])[0]
            team_b_encoded = encoders["team2"].transform([team_b])[0]
            venue_encoded = encoders["venue"].transform([venue])[0]

            # Predict first innings
            X_input = np.array([[team_a_encoded, team_b_encoded, venue_encoded, season_input]])
            pred_runs_first = model.predict(X_input)[0]

            # Simulate second innings
            pred_runs_second = pred_runs_first - np.random.uniform(8, 25)
            win_prob_a = np.clip(50 + (pred_runs_first - pred_runs_second) / 3, 0, 100)
            win_prob_b = 100 - win_prob_a

            # Display results
            st.markdown("### 🎯 Simulation Results")
            c1, c2 = st.columns(2)
            c1.metric(f"{team_a} Predicted Score", f"{int(pred_runs_first)} Runs")
            c2.metric(f"{team_b} Predicted Score", f"{int(pred_runs_second)} Runs")

            # 🏆 Win Probability Chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[team_a, team_b],
                y=[win_prob_a, win_prob_b],
                text=[f"{win_prob_a:.1f}%", f"{win_prob_b:.1f}%"],
                textposition="auto",
                marker_color=["#FFD700", "#1E90FF"]
            ))
            fig.update_layout(
                template="plotly_dark",
                title="🏆 Win Probability (Simulated)",
                xaxis_title="Team",
                yaxis_title="Win Probability (%)",
                plot_bgcolor="#0f1720",
                paper_bgcolor="#0f1720",
                font=dict(color="white")
            )
            st.plotly_chart(fig, width="stretch")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "🛠️ **CRICVISION** — Integrated IPL Analytics  \n"
    "Built with Streamlit + Plotly + SQLAlchemy + MySQL  \n"
    "Modules: Overview, Player Analytics (if ball-level data present), Team Comparison, Venue & Pitch, Partnership Network (if ball-level data present), Predictor."
)
