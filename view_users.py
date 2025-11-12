import pymysql
from pymysql.cursors import DictCursor
import os
import json
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

def show_users():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users;")
            rows = cur.fetchall()

            if not rows:
                print("No users found.")
                return

            print("=== USERS TABLE ===")
            for row in rows:
                print(f"ID: {row['user_id']}")
                print(f"Name: {row['name']}")
                print(f"Email: {row['email']}")
                print(f"Year: {row['year']}")
                print(f"Faculty: {row['faculty']}")
                print(f"Interests: {row['interests']}")
                print(f"End goal: {row['end_goal']}")
                print(f"Timeline: {row['timeline']}")
                print(f"Resume key: {row['resume_key']}")
                print(f"Created at: {row['created_at']}")
                print(f"Updated at: {row['updated_at']}")

                # Handle embedding JSON safely
                embedding = row.get("embedding")
                if embedding:
                    try:
                        parsed = json.loads(embedding) if isinstance(embedding, str) else embedding
                        preview = json.dumps(parsed)[:200]  # Truncate for readability
                        print(f"Embedding (preview): {preview}...")
                    except Exception:
                        print(f"Embedding: {embedding}")
                else:
                    print("Embedding: None")

                print("-" * 60)
    except Exception as e:
        print(f"Error reading users table: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    show_users()
