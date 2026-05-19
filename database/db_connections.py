import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine
from database.db_config import DB_CONFIG

def get_engine():
    """Creates a SQLAlchemy engine for MySQL connection."""
    user = DB_CONFIG['user']
    password = DB_CONFIG['password']
    host = DB_CONFIG['host']
    port = DB_CONFIG['port']
    database = DB_CONFIG['database']
    
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
    print("✅ MySQL Connection Successful!")
    return engine
