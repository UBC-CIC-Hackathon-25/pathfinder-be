import pymysql
from pymysql.cursors import DictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    """Connect to MySQL using env vars."""
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        cursorclass=DictCursor
    )

def drop_users_table():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            print("Dropping `users` table...")
            cur.execute("DROP TABLE IF EXISTS users;")
            conn.commit()
            print("`users` table dropped successfully.")
    except Exception as e:
        print(f"Error dropping users table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    drop_users_table()
