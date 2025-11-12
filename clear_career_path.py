import pymysql
from pymysql.cursors import DictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        cursorclass=DictCursor
    )

def clear_career_path(user_id: str):
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users 
                SET career_path = NULL 
                WHERE user_id = %s
            """, (user_id,))
            conn.commit()

            if cur.rowcount == 0:
                print(f"No user found with ID: {user_id}")
            else:
                print(f"✓ Cleared career_path for user {user_id}")

    except Exception as e:
        print(f"Error updating user: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # 👉 CHANGE THIS to the user_id you want to reset
    clear_career_path("c002d160a44d")
