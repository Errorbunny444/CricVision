🏏 CRICVISION: An Intelligent IPL Analytics & Forecasting System
💡 A Data-Driven Spatiotemporal Performance Analysis and Strategy Simulator for IPL Matches
📘 1. Project Overview

CRICVISION is an advanced IPL analytics system integrating MySQL, Streamlit, and Machine Learning for dynamic, real-time cricket performance insights.

It allows users to:

Explore season-wise, player-wise, and team-wise IPL data

Simulate “What if Team A bats first?” scenarios

Visualize partnership networks and venue stats

Generate PDF reports

Forecast future player performance using time-series modeling

The system serves as a research-grade DBMS + AI-based cricket analysis platform.

🧩 2. Tech Stack
Layer	Tools / Libraries Used
💾 Database	MySQL (Structured Data)
🧠 Graph Analytics	NetworkX / PyVis
🧮 Machine Learning	scikit-learn, Prophet
📊 Visualization	Plotly, Matplotlib
🧱 Backend ORM	SQLAlchemy
💻 Frontend Dashboard	Streamlit (Dark Themed)
🧾 Report Export	pdfkit (wkhtmltopdf)
🧠 Forecasting	Prophet (Seasonal Time-Series)
🗂️ 3. Database Schema

Database: ipl_db

📄 Tables:
Table	Description
matches	Contains match-level info (teams, season, toss, result, venue)
deliveries	Ball-by-ball dataset with runs, wickets, and innings details
match_summary	Aggregated performance data per match
deliveries_clean_backup	Cleaned historical data (for validation/backup)
deliveries_view	MySQL view for analytics joins
🧠 4. System Architecture
           ┌────────────────────┐
           │   MySQL Database    │
           └─────────┬───────────┘
                     │
          SQLAlchemy (ORM Engine)
                     │
           ┌─────────▼───────────┐
           │     Streamlit UI     │
           │  (Interactive Tabs)  │
           └─────────┬───────────┘
                     │
           ┌─────────▼───────────┐
           │  Data Analytics + ML │
           │ Prophet / sklearn    │
           └──────────────────────┘

🧭 5. Modules Overview
🏠 Overview Dashboard

Season-wise stats, matches, venues

Top scoring teams by season

Win percentages and IPL color-coded visuals

Hover-tooltips showing detailed insights

🏏 Player Analytics

ESPN-style dark statcards for batting and bowling

Season-wise trend charts

Performance heatmaps by player

Dynamic top scorers leaderboard

🤝 Team Comparison

Head-to-head stats (matches, wins, runs, wickets)

Comparative performance visualization

🏟 Venue & Pitch Analysis

Venue-wise runs, averages, and economy trends

Filters by season, team, or venue

🧬 Partnership Network

Interactive NetworkX graph visualizing player partnerships

Optimized for visibility in dark mode

🎯 Match Predictor

ML-based outcome predictor using match stats

Shows winning probability

📄 Report Generator

Export Player or Overview Reports as PDF

Uses pdfkit + wkhtmltopdf backend

⏳ Performance Forecasting

Prophet model forecasts player runs for upcoming seasons

Shows predicted trends with upper & lower bounds

🧠 Strategy Simulator

“What if Team A bats first?” analysis

Predicts first innings total using baseline ML model

(Advanced ML version skipped for performance optimization)

⚙️ 6. Setup Instructions
🧱 Step 1: Clone Repository
git clone https://github.com/yourusername/CRICVISION.git
cd CRICVISION

🧩 Step 2: Install Dependencies
pip install -r requirements.txt

📂 Step 3: MySQL Setup
CREATE DATABASE ipl_db;
USE ipl_db;
SOURCE path_to/ipl_tables.sql;

⚙️ Step 4: Update DB Credentials

In app.py:

DB_USER = "root"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "ipl_db"

🧾 Step 5: (Optional) Enable PDF Reports

Install wkhtmltopdf:
📎 Download from here

Then, set in app.py:

config = pdfkit.configuration(wkhtmltopdf=b"path_to_wkhtmltopdf.exe")

▶️ Step 6: Run the App
streamlit run app.py

📊 7. Sample Outputs

✅ Overview graphs with IPL color themes
✅ ESPN-style player cards
✅ Interactive partnership network
✅ Forecast plots (Prophet)
✅ PDF export capability

🔮 8. Future Enhancements
Feature	Description
🌐 Live Data API Integration	Fetch real-time IPL updates from Cricbuzz/ESPN APIs
🧱 Advanced ML Models (XGBoost + SHAP)	Improve prediction accuracy + interpretability
🔒 User Authentication	Streamlit Authenticator for role-based dashboards
🧮 Win Probability Engine	Logistic model predicting match win probability
💬 AI Assistant Module	Natural language query interface for cricket stats
🧾 9. Project Contributors

Author: Aryan Ranadive
Institute: [Your College Name]
Mentor: [Faculty Guide Name]
Course: Database Management Systems (DBMS) Mini Project

🏁 10. Conclusion

CRICVISION demonstrates how data-driven intelligence and relational database design can power a real-world sports analytics engine.
It combines SQL, AI, Graph Theory, and Forecasting in a unified dashboard — making it an ideal model for future IPL strategy systems.

⭐ “Analyze the past. Predict the future. Play smarter — with CRICVISION.”