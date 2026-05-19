import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from database.db_config import DB_CONFIG

def create_database_if_not_exists():
    """Create the database (ipl_db) if it doesn't already exist."""
    user = DB_CONFIG['user']
    password = DB_CONFIG['password']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    db_name = DB_CONFIG['database']

    # First connect to MySQL server without specifying database
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}")
    with engine.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {db_name};"))
        print(f"✅ Database '{db_name}' created or already exists.")

def create_tables():
    """Create tables inside ipl_db."""
    user = DB_CONFIG['user']
    password = DB_CONFIG['password']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    db_name = DB_CONFIG['database']

    # Connect now WITH the database
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}")
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id INT AUTO_INCREMENT PRIMARY KEY,
            season INT,
            match_date DATE,
            city VARCHAR(100),
            venue VARCHAR(150),
            team1 VARCHAR(100),
            team2 VARCHAR(100),
            toss_winner VARCHAR(100),
            toss_decision VARCHAR(20),
            result VARCHAR(50),
            dl_applied TINYINT(1),
            winner VARCHAR(100),
            win_by_runs INT,
            win_by_wickets INT,
            player_of_match VARCHAR(150),
            umpire1 VARCHAR(100),
            umpire2 VARCHAR(100)
        );
        """))
        print("✅ Table 'matches' created successfully inside database.")

if __name__ == "__main__":
    create_database_if_not_exists()
    create_tables()

