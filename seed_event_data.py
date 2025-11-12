from __future__ import annotations

import os
import json
from datetime import date, datetime
from typing import List, Optional, Sequence, Tuple
from contextlib import contextmanager

try:
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "python-dotenv is required. Install it with "
        "`pip install python-dotenv` before running this script."
    ) from exc

try:
    import mysql.connector  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "mysql-connector-python is required. Install it with "
        "`pip install mysql-connector-python` before running this script."
    ) from exc

try:
    import boto3
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "boto3 is required. Install it with "
        "`pip install boto3` before running this script."
    ) from exc


SCHEMA_SQL = """
SET NAMES utf8mb4;
SET time_zone = '+00:00';

DROP TABLE IF EXISTS event_data;

CREATE TABLE event_data (
  event_id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  title             VARCHAR(255)    NOT NULL,
  description       TEXT            NULL,
  enriched_desc     TEXT            NULL,
  start_dt          DATETIME        NOT NULL,
  end_dt            DATETIME        NULL,
  category          VARCHAR(100)    NULL,
  source            VARCHAR(255)    NULL,
  tags              JSON            NULL,
  themes            JSON            NULL,
  target_audience   TEXT            NULL,
  skills_learned    JSON            NULL,
  career_relevance  TEXT            NULL,
  why_attend        TEXT            NULL,
  embedding         JSON            NULL,
  created_at        TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (event_id),
  UNIQUE KEY uk_event_dedupe (title, start_dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
""".strip()


EventRow = Tuple[str, Optional[str], str, Optional[str], Optional[str], Optional[str]]

EVENTS: Sequence[EventRow] = [
    (
        "Challenges, changes, and research: Insights from UBC's University Killam Professors",
        'From provided list; original line began with "ODAY" (interpreted as Nov 12, 2025).',
        "2025-11-12 00:00:00",
        None,
        "Talk",
        "Provided data",
    ),
    (
        "Salary Negotiation",
        None,
        "2025-11-12 12:30:00",
        "2025-11-12 13:30:00",
        "Workshop",
        "Provided data",
    ),
    (
        "Tech Speed Mentoring with Women in Communications and Technology (WCT BC)",
        "Canadian College of Technology and Business, 101 Smithe Street, Vancouver, BC.",
        "2025-11-12 17:30:00",
        "2025-11-12 19:00:00",
        "Mentoring",
        "Provided data",
    ),
    (
        "Ten Surprising Things About the History of Photography in Canada",
        None,
        "2025-11-13 00:00:00",
        None,
        "Talk",
        "Provided data",
    ),
    (
        "[Code-committee] Book Club 2025W: Braiding Sweetgrass",
        "Guided discussion; RSVP link in source.",
        "2025-11-13 15:00:00",
        "2025-11-13 16:00:00",
        "Book Club",
        "Provided data",
    ),
    (
        "BCSSA Annual General Meeting (AGM)",
        "Student society AGM; free merch for attendees.",
        "2025-11-13 17:30:00",
        "2025-11-13 18:30:00",
        "Meeting",
        "Provided data",
    ),
    (
        "Doctoral Oral Defence - Rohit Murali",
        None,
        "2025-11-13 12:30:00",
        "2025-11-13 15:30:00",
        "Defense",
        "Provided data",
    ),
    (
        "AI & the Future of Medicine: Bridging AI Innovation and Health Equity",
        "1.5-day event (Sat half-day, Sun full day). UBC Life Sciences Centre, 2350 Health Sciences Mall.",
        "2025-11-15 00:00:00",
        "2025-11-16 23:59:59",
        "Conference",
        "Provided data",
    ),
    (
        "Climate Emergency Week",
        None,
        "2025-11-17 00:00:00",
        "2025-11-21 23:59:59",
        "Thematic Week",
        "Provided data",
    ),
    (
        "Financial Literacy for Students: Presented by Y Foundation in Partnership with CFEE and Gentai Capital Corporation",
        None,
        "2025-11-17 17:30:00",
        "2025-11-17 19:30:00",
        "Workshop",
        "Provided data",
    ),
    (
        "Forestry Info Session - Master of International Forestry",
        None,
        "2025-11-17 00:00:00",
        None,
        "Info Session",
        "Provided data",
    ),
    (
        "Purpose-Driven: UBC Entrepreneurs Innovating in a Changing World",
        None,
        "2025-11-18 17:30:00",
        "2025-11-18 20:00:00",
        "Panel",
        "Provided data",
    ),
    (
        "WICS x SAP: Day in the Life Panel & Office Tour",
        "SAP Office, 910 Mainland St, Vancouver, BC V6B 1A9. RSVP required.",
        "2025-11-18 15:00:00",
        "2025-11-18 18:00:00",
        "Panel / Tour",
        "Provided data",
    ),
    (
        "AccessCS: Social + Sushi + Origami Night! (RSVP by Nov 18)",
        "Undergraduate Lounge (where Pho Real is). Space limited.",
        "2025-11-20 17:00:00",
        "2025-11-20 19:00:00",
        "Social",
        "Provided data",
    ),
    (
        "Folding Furoshiki for Sustainable Gift-Giving",
        None,
        "2025-11-19 00:00:00",
        None,
        "Workshop",
        "Provided data",
    ),
    (
        "AI Safety Technical Reading Group - Scalable Oversight",
        "Every other Wednesday series; snacks provided. Location: IKB (room TBD).",
        "2025-11-19 17:00:00",
        "2025-11-19 19:00:00",
        "Reading Group",
        "Provided data",
    ),
    (
        "TikTok @ UBC Info Session: Unlock Your Leadership Potential",
        "Virtual session; recruiting for full-time and Summer Internship programs.",
        "2025-11-19 17:00:00",
        "2025-11-19 18:00:00",
        "Info Session",
        "Provided data",
    ),
    (
        "Employer On-Campus: Recruitment of Policy Leaders Program (Government of Canada)",
        None,
        "2025-11-19 18:00:00",
        "2025-11-19 19:00:00",
        "Info Session",
        "Provided data",
    ),
    (
        "UBC BizTech KickStart Startup Competition",
        "Week-long competition from kickoff to demo days.",
        "2025-11-19 00:00:00",
        "2025-11-26 23:59:59",
        "Competition",
        "Provided data",
    ),
    (
        "Student Earthquake and Emergency Preparedness Workshop",
        None,
        "2025-11-20 00:00:00",
        None,
        "Workshop",
        "Provided data",
    ),
    (
        'Opening Celebrations of "Entangled Territories: Tibet Through Images"',
        None,
        "2025-11-20 00:00:00",
        None,
        "Opening / Exhibit",
        "Provided data",
    ),
    (
        "Forestry Info Session - Master of Geomatics for Environmental Management",
        None,
        "2025-11-20 00:00:00",
        None,
        "Info Session",
        "Provided data",
    ),
    (
        "Forestry Info Session - Dual Degree, Master of Forestry in Green Business",
        None,
        "2025-11-21 00:00:00",
        None,
        "Info Session",
        "Provided data",
    ),
    (
        "Planning for Healthy Communities with Sa̱nala Planning and Cwelcwélt Consulting",
        None,
        "2025-11-25 00:00:00",
        None,
        "Talk",
        "Provided data",
    ),
    (
        "Disability Justice Book Club: Disabled and Proud Cohort",
        None,
        "2025-11-27 00:00:00",
        None,
        "Book Club",
        "Provided data",
    ),
    (
        "Forestry Info Session - Master of Sustainable Forest Management",
        None,
        "2025-11-27 00:00:00",
        None,
        "Info Session",
        "Provided data",
    ),
    (
        "Forestry Info Session - Master of Urban Forestry Leadership",
        None,
        "2025-11-28 00:00:00",
        None,
        "Info Session",
        "Provided data",
    ),
    (
        "14 Not Forgotten",
        None,
        "2025-12-01 00:00:00",
        None,
        "Memorial",
        "Provided data",
    ),
    (
        "Getting the Most Out of Your Academic Experience: Designing Your Grad School/Career Strategy",
        None,
        "2025-12-03 00:00:00",
        None,
        "Workshop",
        "Provided data",
    ),
    (
        "AI Safety Technical Reading Group - Control",
        "Location: IKB (room TBD).",
        "2025-12-03 17:00:00",
        "2025-12-03 19:00:00",
        "Reading Group",
        "Provided data",
    ),
    (
        "PhD Thesis Defence - Chenwei Zhang",
        None,
        "2025-12-15 09:30:00",
        "2025-12-15 13:00:00",
        "Defense",
        "Provided data",
    ),
    (
        "AI Safety Technical Reading Group - Constitutional AI",
        "Location: IKB (room TBD).",
        "2025-12-17 17:00:00",
        "2025-12-17 19:00:00",
        "Reading Group",
        "Provided data",
    ),
    (
        "Disability Justice Book Club: Disabled and Proud Cohort",
        None,
        "2025-12-18 00:00:00",
        None,
        "Book Club",
        "Provided data",
    ),
    (
        "PhD Thesis Defense - Mir Rayat Imtiaz Hossain",
        None,
        "2025-10-06 12:00:00",
        "2025-10-06 16:00:00",
        "Defense",
        "Provided data",
    ),
    (
        "Fireside Insights: Navigating the Landscape of Artificial Intelligence",
        "HSBC Hall, UBC Robson Square (800 Robson St, Vancouver, BC V6Z 3B7).",
        "2025-10-09 17:00:00",
        "2025-10-09 18:15:00",
        "Talk",
        "Provided data",
    ),
    (
        "MSc Essay Presentation - Jason Hall",
        None,
        "2025-09-26 11:00:00",
        "2025-09-26 12:00:00",
        "Presentation",
        "Provided data",
    ),
    (
        "Treat Yourself UBC Student Giveaway",
        "Contest runs Nov 1-30, 2025; open to registered UBC students.",
        "2025-11-01 00:00:00",
        "2025-11-30 23:59:59",
        "Campaign",
        "Provided data",
    ),
]


def get_bedrock_client():
    """Initialize and return AWS Bedrock client"""
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )


def enrich_event_with_llm(event_data: dict) -> dict:
    """
    Use Claude to generate comprehensive event metadata
    Returns: enriched event data with tags, themes, target audience, etc.
    """
    bedrock = get_bedrock_client()
    
    prompt = f"""Given this UBC event, generate comprehensive metadata to help students discover relevant opportunities.

Event Title: {event_data['title']}
Category: {event_data.get('category', 'N/A')}
Description: {event_data.get('description', 'No description provided')}

Generate the following in STRICT JSON format (no markdown, no extra text):

{{
  "enriched_description": "2-3 sentence engaging description of the event",
  "tags": ["tag1", "tag2", ...],  // 5-8 relevant keywords/topics
  "themes": ["theme1", "theme2", ...],  // 3-5 overarching themes
  "target_audience": "Who should attend this event",
  "skills_learned": ["skill1", "skill2", ...],  // 3-6 skills/takeaways
  "career_relevance": "How this helps career development (1-2 sentences)",
  "why_attend": "Compelling reasons to attend (2-3 sentences)"
}}

Be specific and relevant to UBC students. Focus on career development, networking, skill-building, and academic growth."""

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.3,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",  # Using Haiku for speed
            body=body
        )
        
        response_body = json.loads(response["body"].read())
        text = response_body["content"][0]["text"].strip()
        
        # Parse JSON response
        enriched = json.loads(text)
        return enriched
        
    except Exception as e:
        print(f"  ⚠️  LLM enrichment failed: {e}")
        # Return fallback values
        return {
            "enriched_description": event_data.get('description', 'A valuable UBC event opportunity.'),
            "tags": [event_data.get('category', 'general')],
            "themes": ["professional development"],
            "target_audience": "UBC students",
            "skills_learned": ["networking", "knowledge"],
            "career_relevance": "Relevant for career growth and professional development.",
            "why_attend": "Great opportunity to learn and connect with others."
        }


def create_enriched_event_embedding(event_data: dict, enriched_metadata: dict) -> list:
    """
    Create embedding from ALL event data including LLM-enriched metadata
    """
    bedrock = get_bedrock_client()
    
    # Combine EVERYTHING into the embedding text
    text_to_embed = f"""
Event: {event_data['title']}
Category: {event_data.get('category', 'General')}

Description: {event_data.get('description', 'No description')}
Enriched Description: {enriched_metadata.get('enriched_description', '')}

Tags: {', '.join(enriched_metadata.get('tags', []))}
Themes: {', '.join(enriched_metadata.get('themes', []))}
Target Audience: {enriched_metadata.get('target_audience', 'UBC students')}
Skills Learned: {', '.join(enriched_metadata.get('skills_learned', []))}
Career Relevance: {enriched_metadata.get('career_relevance', '')}
Why Attend: {enriched_metadata.get('why_attend', '')}

This event is ideal for students interested in: {', '.join(enriched_metadata.get('tags', []))}
Students will gain: {', '.join(enriched_metadata.get('skills_learned', []))}
""".strip()
    
    try:
        request_body = json.dumps({
            "inputText": text_to_embed
        })
        
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        
        return embedding
        
    except Exception as e:
        print(f"  ✗ Failed to create embedding: {str(e)}")
        return None


@contextmanager
def get_db_connection():
    """Context manager for MySQL database connections"""
    conn = None
    try:
        ssl_config = {}
        if os.getenv("DB_SSL_CA"):
            ssl_config["ssl_ca"] = os.getenv("DB_SSL_CA")
        
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "events"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD"),
            port=int(os.getenv("DB_PORT", "3306")),
            autocommit=True,
            use_pure=True,
            **ssl_config
        )
        yield conn
        conn.commit()
    except mysql.connector.Error as err:
        if conn:
            conn.rollback()
        raise SystemExit(f"Unable to connect to MySQL: {err}") from err
    finally:
        if conn:
            conn.close()


def apply_schema(cursor: mysql.connector.cursor.MySQLCursor) -> None:
    for statement in SCHEMA_SQL.split(";"):
        stmt = statement.strip()
        if stmt:
            cursor.execute(stmt)


def insert_events(cursor: mysql.connector.cursor.MySQLCursor, events: Sequence[EventRow]) -> None:
    payload: List[Tuple[str, Optional[str], Optional[str], datetime, Optional[datetime], 
                        Optional[str], Optional[str], Optional[str], Optional[str], 
                        Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]] = []
    
    for title, description, start_dt, end_dt, category, source in events:
        print(f"\n{'='*70}")
        print(f"Processing: {title[:60]}...")
        
        # Create event data dict
        event_data = {
            'title': title,
            'description': description,
            'start_dt': start_dt,
            'end_dt': end_dt,
            'category': category
        }
        
        # Step 1: Enrich with LLM
        print(f"  → Generating metadata with Claude...")
        enriched_metadata = enrich_event_with_llm(event_data)
        
        # Step 2: Create embedding from enriched data
        print(f"  → Creating embedding...")
        embedding = create_enriched_event_embedding(event_data, enriched_metadata)
        embedding_json = json.dumps(embedding) if embedding else None
        
        # Prepare data for insertion
        payload.append((
            title,
            description,
            enriched_metadata.get('enriched_description'),
            datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S"),
            datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S") if end_dt else None,
            category,
            source,
            json.dumps(enriched_metadata.get('tags', [])),
            json.dumps(enriched_metadata.get('themes', [])),
            enriched_metadata.get('target_audience'),
            json.dumps(enriched_metadata.get('skills_learned', [])),
            enriched_metadata.get('career_relevance'),
            enriched_metadata.get('why_attend'),
            embedding_json,
        ))
        
        print(f"  ✓ Complete!")

    insert_sql = """
        INSERT IGNORE INTO event_data
        (title, description, enriched_desc, start_dt, end_dt, category, source, 
         tags, themes, target_audience, skills_learned, career_relevance, why_attend, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.executemany(insert_sql, payload)
    print(f"\n{'='*70}")
    print(f"✓ Inserted {len(payload)} fully enriched events with embeddings")


def fetch_all_events(cursor: mysql.connector.cursor.MySQLCursor) -> List[dict]:
    cursor.execute(
        """
        SELECT event_id, title, description, enriched_desc, start_dt, end_dt, 
               category, source, tags, themes, target_audience, skills_learned, 
               career_relevance, why_attend, embedding, created_at
        FROM event_data
        ORDER BY start_dt, event_id
        """
    )
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    results: List[dict] = []
    for row in rows:
        record = {}
        for column, value in zip(columns, row):
            if isinstance(value, (datetime, date)):
                record[column] = value.strftime("%Y-%m-%d %H:%M:%S")
            else:
                record[column] = value
        results.append(record)
    return results


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            print("Creating schema...")
            apply_schema(cursor)
            print("✓ Schema created\n")
            
            print("Enriching events with LLM and creating embeddings...")
            insert_events(cursor, EVENTS)
            
            # Optionally print all events if PRINT_EVENTS env var is set
            if os.getenv("PRINT_EVENTS", "").lower() in ("true", "1", "yes"):
                rows = fetch_all_events(cursor)
                print(json.dumps(rows, indent=2))
        finally:
            cursor.close()


if __name__ == "__main__":
    main()