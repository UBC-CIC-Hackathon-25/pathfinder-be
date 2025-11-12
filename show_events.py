import os
import json
import mysql.connector
from mysql.connector.cursor import MySQLCursorDict
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Connect to MySQL using env vars."""
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


def _parse_json_field(value):
    if not value:
        return None
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value  # fallback raw


def show_events():
    try:
        conn = get_connection()
        cursor: MySQLCursorDict = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM event_data ORDER BY start_dt, event_id;")
        rows = cursor.fetchall()

        if not rows:
            print("No events found.")
            return

        print("=" * 100)
        print(f"SHOWING ALL {len(rows)} EVENTS FROM event_data")
        print("=" * 100)
        print()

        for idx, row in enumerate(rows, 1):
            print(f"[{idx}/{len(rows)}] Event ID: {row['event_id']}")
            print(f"Title: {row['title']}")
            print(f"Category: {row.get('category') or 'N/A'}")
            print(f"Source: {row.get('source') or 'N/A'}")
            print(f"Start: {row.get('start_dt')}")
            print(f"End: {row.get('end_dt') or 'N/A'}")
            print(f"Created at: {row.get('created_at')}")

            print("\n--- Descriptions ---")
            print(f"Raw description: {row.get('description') or 'N/A'}")
            print(f"Enriched desc  : {row.get('enriched_desc') or 'N/A'}")

            print("\n--- Metadata ---")
            tags = _parse_json_field(row.get("tags"))
            themes = _parse_json_field(row.get("themes"))
            skills = _parse_json_field(row.get("skills_learned"))

            print(f"Tags           : {tags if tags else 'N/A'}")
            print(f"Themes         : {themes if themes else 'N/A'}")
            print(f"Target audience: {row.get('target_audience') or 'N/A'}")
            print(f"Skills learned : {skills if skills else 'N/A'}")
            print(f"Career relevance: {row.get('career_relevance') or 'N/A'}")
            print(f"Why attend     : {row.get('why_attend') or 'N/A'}")

            print("\n--- Embedding ---")
            embedding_raw = row.get("embedding")
            if embedding_raw:
                try:
                    parsed = (
                        json.loads(embedding_raw)
                        if isinstance(embedding_raw, str)
                        else embedding_raw
                    )
                    if isinstance(parsed, list):
                        print(f"✓ Embedding: {len(parsed)} dimensions")
                    else:
                        print(f"✓ Embedding (non-list type): {type(parsed)}")
                except Exception as e:
                    print(f"✗ Embedding parse error: {e}")
            else:
                print("✗ Embedding: None")

            print("-" * 100)

    except Exception as e:
        print(f"Error reading event_data table: {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


def show_events_summary():
    """Show a concise summary of all events."""
    try:
        conn = get_connection()
        cursor: MySQLCursorDict = conn.cursor(dictionary=True)

        cursor.execute("SELECT * FROM event_data ORDER BY start_dt, event_id;")
        rows = cursor.fetchall()

        if not rows:
            print("No events found.")
            return

        print("=" * 100)
        print(f"EVENT SUMMARY ({len(rows)} total events in event_data)")
        print("=" * 100)

        # Count by category
        categories = {}
        with_embeddings = 0
        for row in rows:
            cat = row.get("category") or "Uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
            if row.get("embedding"):
                with_embeddings += 1

        print("\nEvents by Category:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count} events")

        print(f"\nEvents with embeddings: {with_embeddings}/{len(rows)}")

        print("\n" + "=" * 100)
        print(f"{'Emb':<4} {'ID':<5} {'Category':<22} {'Start':<19} {'Title'}")
        print("-" * 100)

        for row in rows:
            event_id = row["event_id"]
            category = (row.get("category") or "N/A")[:20]
            start = str(row.get("start_dt"))[:19]
            title = row["title"]
            title = title[:45] + "..." if len(title) > 48 else title

            has_emb = "✓" if row.get("embedding") else "✗"
            print(f"{has_emb:<4} {event_id:<5} {category:<22} {start:<19} {title}")

    except Exception as e:
        print(f"Error reading event_data table: {e}")
    finally:
        if "cursor" in locals():
            cursor.close()
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        show_events_summary()
    else:
        show_events()
        print("\nTip: Run `python show_events.py summary` for a compact view.")
