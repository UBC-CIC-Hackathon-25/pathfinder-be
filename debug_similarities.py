import os
import json
import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        cursorclass=DictCursor,
    )

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def debug_user_event_similarities(user_id: str):
    """Show similarity scores between a user and all events"""
    conn = get_connection()
    
    try:
        # Get user embedding
        with conn.cursor() as cur:
            cur.execute("""
                SELECT name, faculty, embedding
                FROM users
                WHERE user_id = %s
            """, (user_id,))
            
            user = cur.fetchone()
            
            if not user:
                print(f"❌ User {user_id} not found")
                return
            
            if not user['embedding']:
                print(f"❌ User {user['name']} has no embedding")
                return
            
            user_emb = np.array(json.loads(user['embedding']), dtype=np.float32)
            print(f"✓ User: {user['name']} ({user['faculty']})")
            print(f"✓ User embedding dimensions: {len(user_emb)}")
            print(f"\n{'='*80}\n")
            
        # Get all events with embeddings
        with conn.cursor() as cur:
            cur.execute("""
                SELECT event_id, title, category, embedding
                FROM event_data
                ORDER BY event_id
            """)
            
            events = cur.fetchall()
            
            if not events:
                print("❌ No events found")
                return
            
            print(f"Found {len(events)} events\n")
            
            # Calculate similarities
            scored = []
            for event in events:
                if not event['embedding']:
                    print(f"⚠️  Event {event['event_id']} has no embedding")
                    continue
                
                event_emb = np.array(json.loads(event['embedding']), dtype=np.float32)
                sim = cosine_sim(user_emb, event_emb)
                scored.append((sim, event))
            
            # Sort by similarity
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Print results
            print("SIMILARITY SCORES (sorted high to low):")
            print(f"{'Score':<8} {'ID':<6} {'Category':<20} {'Title'}")
            print("="*80)
            
            for sim, event in scored:
                marker = "🔥" if sim >= 0.30 else "✓" if sim >= 0.25 else "○" if sim >= 0.20 else "·"
                title = event['title'][:50] + "..." if len(event['title']) > 50 else event['title']
                category = (event['category'] or "N/A")[:18]
                print(f"{marker} {sim:.4f}  {event['event_id']:<6} {category:<20} {title}")
            
            # Summary
            print(f"\n{'='*80}\n")
            above_30 = sum(1 for sim, _ in scored if sim >= 0.30)
            above_25 = sum(1 for sim, _ in scored if sim >= 0.25)
            above_20 = sum(1 for sim, _ in scored if sim >= 0.20)
            above_15 = sum(1 for sim, _ in scored if sim >= 0.15)
            
            print(f"SUMMARY:")
            print(f"  🔥 Similarity >= 0.30: {above_30} events")
            print(f"  ✓  Similarity >= 0.25: {above_25} events (current threshold)")
            print(f"  ○  Similarity >= 0.20: {above_20} events")
            print(f"  ·  Similarity >= 0.15: {above_15} events")
            print(f"\n  Max similarity: {scored[0][0]:.4f}")
            print(f"  Min similarity: {scored[-1][0]:.4f}")
            print(f"  Average similarity: {np.mean([s for s, _ in scored]):.4f}")
            
            if above_25 == 0:
                print(f"\n⚠️  NO EVENTS above threshold 0.25!")
                print(f"   Consider lowering min_sim to 0.15 or 0.20")
                print(f"   OR add more relevant CS/tech/AI events to your database")
            
    finally:
        conn.close()

if __name__ == "__main__":
    # Debug Ryan's similarities
    debug_user_event_similarities("c002d160a44d")