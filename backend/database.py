import os
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "/app/data/attendance.db")

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
