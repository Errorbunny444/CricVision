import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests/test_db_connection.py
from sqlalchemy import create_engine, text
import pandas as pd

DB_USER = "root"
DB_PASSWORD = "Dinner!18"
DB_HOST = "localhost"
DB_NAME = "ipl_db"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

# 1) basic count (use safe alias)
df = pd.read_sql("SELECT COUNT(*) AS cnt FROM deliveries", engine)
print("deliveries rows:", df['cnt'].iloc[0])

# 2) show tables available
tables = pd.read_sql("SHOW TABLES", engine)
print("tables in DB:")
print(tables)

# 3) peek a few rows from deliveries (avoid huge fetch)
peek = pd.read_sql("SELECT match_id, batter, batsman_runs FROM deliveries LIMIT 5", engine)
print(peek)

