"""
prepare_deliveries_clean.py

Creates a deduplicated, inning-fixed table `deliveries_clean` from `deliveries`.

Requirements: your project already provides `database.db_connections.get_engine()` returning SQLAlchemy engine.
Run:
    (venv) python src/prepare_deliveries_clean.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine

# 1️⃣ MySQL connection details
DB_URI = "mysql+pymysql://root:Dinner!18@localhost:3306/ipl_db"
engine = create_engine(DB_URI)

# 2️⃣ Path to the ORIGINAL IPL CSV (replace with your actual file path)
csv_path = r"E:\SY\DBMS\CRICVISION\data\IPL.csv"

print("✅ MySQL Connection Successful!")
print("📂 Loading from CSV...")

# 3️⃣ Load raw CSV exactly as-is
df = pd.read_csv(csv_path, low_memory=False)

print(f"✅ CSV loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 4️⃣ Normalize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 5️⃣ Write clean version into MySQL
table_name = "deliveries"
df.to_sql(table_name, engine, if_exists="replace", index=False)

print(f"✅ Uploaded to MySQL as `{table_name}` — total {len(df):,} rows")

