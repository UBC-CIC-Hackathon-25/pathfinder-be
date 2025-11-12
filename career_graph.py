# career_graph.py - Enhanced version with better visualization

import os
import json
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
import pymysql
from pymysql.cursors import DictCursor
from dotenv import load_dotenv

import boto3

load_dotenv()

# =========================
# DB CONNECTION
# =========================

def get_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        port=int(os.getenv("DB_PORT", 3306)),
        cursorclass=DictCursor,
    )


# =========================
# EMBEDDING HELPERS
# =========================

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================
# DATA ACCESS
# =========================

def get_user_embedding_and_profile(user_id: str) -> Tuple[np.ndarray, Dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    embedding,
                    name,
                    faculty,
                    year,
                    interests,
                    end_goal,
                    timeline
                FROM users
                WHERE user_id = %s
                """,
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"User {user_id} not found")

            emb = np.array(json.loads(row["embedding"]), dtype=np.float32)

            profile = {
                "user_id": user_id,
                "name": row.get("name"),
                "faculty": row.get("faculty"),
                "year": row.get("year"),
                "interests": row.get("interests"),
                "end_goal": row.get("end_goal"),
                "timeline": row.get("timeline"),
            }

            return emb, profile
    finally:
        conn.close()


def get_events_with_all_data() -> List[Dict]:
    """Fetch ALL event data including enriched metadata"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    event_id,
                    title,
                    description,
                    enriched_desc,
                    start_dt,
                    end_dt,
                    category,
                    source,
                    tags,
                    themes,
                    target_audience,
                    skills_learned,
                    career_relevance,
                    why_attend,
                    embedding
                FROM event_data
                WHERE embedding IS NOT NULL
                """
            )
            rows = cur.fetchall()

            for r in rows:
                # Parse JSON fields
                if r.get('tags'):
                    r['tags'] = json.loads(r['tags'])
                if r.get('themes'):
                    r['themes'] = json.loads(r['themes'])
                if r.get('skills_learned'):
                    r['skills_learned'] = json.loads(r['skills_learned'])
                if r.get('embedding'):
                    r['embedding'] = np.array(json.loads(r['embedding']), dtype=np.float32)
                
                # Format dates for display
                if r.get('start_dt'):
                    r['start_dt_display'] = r['start_dt'].strftime("%b %d, %Y %I:%M %p")
                if r.get('end_dt'):
                    r['end_dt_display'] = r['end_dt'].strftime("%b %d, %Y %I:%M %p")

            return rows
    finally:
        conn.close()


def rag_retrieve_events_for_user(
    user_id: str, k: int = 25, min_sim: float = 0.15
) -> Tuple[List[Dict], Dict]:
    """
    RAG-style retrieval with comprehensive event data
    """
    user_emb, profile = get_user_embedding_and_profile(user_id)
    events = get_events_with_all_data()

    scored: List[Tuple[float, Dict]] = []
    for ev in events:
        sim = cosine_sim(user_emb, ev["embedding"])
        if sim >= min_sim:
            scored.append((sim, ev))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_events = []
    for sim, ev in scored[:k]:
        ev_copy = ev.copy()
        ev_copy["similarity"] = round(sim, 4)
        # Keep embedding for potential use but convert to list
        if 'embedding' in ev_copy:
            ev_copy.pop('embedding')
        top_events.append(ev_copy)

    return top_events, profile


# =========================
# LLM PROMPT / CALL
# =========================

GRAPH_SPEC_SYSTEM_MSG = """
You are PathFinder UBC, an AI career roadmap architect for UBC students.

You will receive:
- Student profile: name, faculty, year, interests, end goal, timeline
- Pre-filtered relevant events with rich metadata (description, tags, themes, skills, career relevance)

Your task: Build a clear, actionable career roadmap as a directed graph.

REQUIREMENTS:

1. **Stages (3-6 stages)**:
   - Create logical progression stages from student's current position to their end goal
   - Examples: "Foundation Building", "Skill Development", "Networking", "Specialization", "Goal Achievement"
   - Each stage should have:
     * Unique id (lowercase, no spaces, e.g., "foundation_building")
     * Clear label (human-readable)
     * Description (2-3 sentences explaining this phase)
     * List of relevant event_ids
     * must_attend_event_ids (1-3 critical events per stage)

2. **Event Selection**:
   - Only include events that directly help achieve the student's end_goal
   - Consider faculty, year level, and timeline
   - Prioritize events with higher similarity scores for must_attend
   - Balance different event types (workshops, info sessions, networking, etc.)

3. **Stage Ordering**:
   - Create a logical flow from current state → end goal
   - Early stages: exploration, foundation building
   - Middle stages: skill development, networking
   - Late stages: specialization, goal-specific preparation

4. **Edges**:
   - Connect stages in sequence
   - Each edge needs: from_stage, to_stage, label (reason for progression)
   - Must form a directed acyclic graph (DAG)

OUTPUT FORMAT (STRICT JSON, no markdown):

{
  "stages": [
    {
      "id": "stage_id",
      "label": "Stage Name",
      "description": "What this stage accomplishes and why it matters",
      "event_ids": [1, 2, 3, 4],
      "must_attend_event_ids": [1, 2]
    }
  ],
  "edges": [
    {
      "from_stage": "stage1_id",
      "to_stage": "stage2_id",
      "label": "Progression reason"
    }
  ]
}

CRITICAL: Only use event_ids that exist in the provided events list.
"""


def build_llm_input(profile: Dict, events: List[Dict]) -> str:
    """Build comprehensive input for LLM with all event metadata"""
    payload = {
        "student_profile": {
            "name": profile.get("name"),
            "faculty": profile.get("faculty"),
            "year": profile.get("year"),
            "interests": profile.get("interests"),
            "end_goal": profile.get("end_goal"),
            "timeline": profile.get("timeline")
        },
        "events": []
    }

    for ev in events:
        event_summary = {
            "event_id": ev["event_id"],
            "title": ev["title"],
            "category": ev.get("category"),
            "description": ev.get("description"),
            "enriched_description": ev.get("enriched_desc"),
            "start_date": ev.get("start_dt_display"),
            "end_date": ev.get("end_dt_display"),
            "tags": ev.get("tags", []),
            "themes": ev.get("themes", []),
            "target_audience": ev.get("target_audience"),
            "skills_learned": ev.get("skills_learned", []),
            "career_relevance": ev.get("career_relevance"),
            "why_attend": ev.get("why_attend"),
            "similarity_score": ev.get("similarity")
        }
        payload["events"].append(event_summary)

    return json.dumps(payload, ensure_ascii=False, indent=2)


bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-west-2"))


def call_llm_for_graph(profile: Dict, events: List[Dict]) -> Dict:
    """Calls Claude via Bedrock to generate the roadmap"""
    user_json = build_llm_input(profile, events)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "temperature": 0.2,
            "system": GRAPH_SPEC_SYSTEM_MSG,
            "messages": [
                {
                    "role": "user",
                    "content": user_json
                }
            ]
        }
    )

    response = bedrock.invoke_model(
        modelId=os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"
        ),
        body=body,
    )

    resp_body = json.loads(response["body"].read())

    if "content" in resp_body:
        text = ""
        for block in resp_body["content"]:
            if block.get("type") == "text":
                text += block.get("text", "")
    elif "outputText" in resp_body:
        text = resp_body["outputText"]
    else:
        raise RuntimeError(f"Unexpected Bedrock response format: {resp_body}")

    try:
        roadmap = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {text}")
        raise e

    return roadmap


# =========================
# REACT FLOW CONVERSION (Enhanced)
# =========================

def roadmap_to_react_flow(roadmap: Dict, profile: Dict, events_lookup: Dict[int, Dict]) -> Dict:
    """
    Convert roadmap to React Flow with enhanced visual layout
    Creates: User Node → Stage Nodes with Events → Goal Node
    """
    nodes: List[Dict] = []
    edges: List[Dict] = []

    # UPDATED Layout configuration for better spacing
    STAGE_WIDTH = 400
    STAGE_X_SPACING = 650       # Increased from 500
    EVENT_Y_SPACING = 250        # Increased from 200
    STAGE_EVENT_GAP = 200        # New - gap between stage and first event
    START_X = 150                # Increased from 100
    START_Y = 400                # Increased from 300

    # ============================================
    # 1. CREATE USER START NODE
    # ============================================
    user_node_id = "user-start"
    nodes.append({
        "id": user_node_id,
        "type": "input",  # Special start node type
        "data": {
            "label": profile.get("name", "Student"),
            "type": "user",
            "faculty": profile.get("faculty"),
            "year": profile.get("year"),
            "interests": profile.get("interests"),
            "timeline": profile.get("timeline")
        },
        "position": {"x": START_X, "y": START_Y},
        "style": {
            "background": "#667eea",
            "color": "white",
            "border": "2px solid #764ba2",
            "borderRadius": "12px",
            "padding": "20px",
            "width": 250
        }
    })

    # ============================================
    # 2. CREATE STAGE NODES & EVENT NODES
    # ============================================
    stages = roadmap.get("stages", [])
    
    for stage_idx, stage in enumerate(stages):
        stage_id = stage["id"]
        stage_node_id = f"stage-{stage_id}"
        
        # Calculate stage position
        stage_x = START_X + (stage_idx + 1) * STAGE_X_SPACING
        stage_y = 100
        
        # Create stage group node
        nodes.append({
            "id": stage_node_id,
            "type": "default",
            "data": {
                "label": stage["label"],
                "description": stage.get("description", ""),
                "type": "stage",
                "stageIndex": stage_idx + 1,
                "totalStages": len(stages)
            },
            "position": {"x": stage_x, "y": stage_y},
            "style": {
                "background": "#f7fafc",
                "border": "2px solid #e2e8f0",
                "borderRadius": "8px",
                "padding": "15px",
                "minWidth": STAGE_WIDTH,
                "fontSize": "14px"
            }
        })
        
        # Add edge from previous stage (or user) to this stage
        if stage_idx == 0:
            # Connect user to first stage
            edges.append({
                "id": f"{user_node_id}->{stage_node_id}",
                "source": user_node_id,
                "target": stage_node_id,
                "type": "smoothstep",
                "animated": True,
                "style": {"stroke": "#667eea", "strokeWidth": 2},
                "label": "Begin Journey"
            })
        
        # Create event nodes within stage
        event_ids = stage.get("event_ids", [])
        must_attend = set(stage.get("must_attend_event_ids", []))
        
        for event_idx, event_id in enumerate(event_ids):
            event_data = events_lookup.get(event_id)
            if not event_data:
                continue
                
            event_node_id = f"event-{event_id}"
            is_must_attend = event_id in must_attend
            
            # Calculate event position (stacked vertically within stage column)
            event_x = stage_x
            event_y = stage_y + STAGE_EVENT_GAP + (event_idx * EVENT_Y_SPACING)  # UPDATED
            
            # Create event node with rich data
            nodes.append({
                "id": event_node_id,
                "type": "default",
                "data": {
                    "label": event_data.get("title", f"Event {event_id}"),
                    "type": "event",
                    "eventId": event_id,
                    "category": event_data.get("category"),
                    "description": event_data.get("enriched_desc") or event_data.get("description"),
                    "startDate": event_data.get("start_dt_display"),
                    "endDate": event_data.get("end_dt_display"),
                    "tags": event_data.get("tags", []),
                    "skills": event_data.get("skills_learned", []),
                    "whyAttend": event_data.get("why_attend"),
                    "careerRelevance": event_data.get("career_relevance"),
                    "isMustAttend": is_must_attend,
                    "similarity": event_data.get("similarity")
                },
                "position": {"x": event_x, "y": event_y},
                "style": {
                    "background": "#fef5e7" if is_must_attend else "white",
                    "border": f"2px solid {'#f39c12' if is_must_attend else '#cbd5e0'}",
                    "borderRadius": "8px",
                    "padding": "12px",
                    "width": STAGE_WIDTH - 50,
                    "fontSize": "12px"
                }
            })
            
            # Connect stage to event
            edges.append({
                "id": f"{stage_node_id}->{event_node_id}",
                "source": stage_node_id,
                "target": event_node_id,
                "type": "smoothstep",
                "style": {
                    "stroke": "#f39c12" if is_must_attend else "#cbd5e0",
                    "strokeWidth": 2 if is_must_attend else 1
                }
            })
    
    # ============================================
    # 3. ADD STAGE-TO-STAGE EDGES
    # ============================================
    for edge_spec in roadmap.get("edges", []):
        from_id = f"stage-{edge_spec['from_stage']}"
        to_id = f"stage-{edge_spec['to_stage']}"
        
        edges.append({
            "id": f"{from_id}->{to_id}",
            "source": from_id,
            "target": to_id,
            "type": "smoothstep",
            "animated": True,
            "label": edge_spec.get("label", ""),
            "style": {"stroke": "#48bb78", "strokeWidth": 2.5},
            "labelStyle": {"fill": "#48bb78", "fontWeight": 600}
        })
    
    # ============================================
    # 4. CREATE GOAL END NODE
    # ============================================
    goal_node_id = "user-goal"
    goal_x = START_X + (len(stages) + 1) * STAGE_X_SPACING
    goal_y = START_Y
    
    nodes.append({
        "id": goal_node_id,
        "type": "output",  # Special end node type
        "data": {
            "label": "Goal Achievement",
            "type": "goal",
            "goal": profile.get("end_goal"),
            "timeline": profile.get("timeline")
        },
        "position": {"x": goal_x, "y": goal_y},
        "style": {
            "background": "#48bb78",
            "color": "white",
            "border": "2px solid #38a169",
            "borderRadius": "12px",
            "padding": "20px",
            "width": 250
        }
    })
    
    # Connect last stage to goal
    if stages:
        last_stage_id = f"stage-{stages[-1]['id']}"
        edges.append({
            "id": f"{last_stage_id}->{goal_node_id}",
            "source": last_stage_id,
            "target": goal_node_id,
            "type": "smoothstep",
            "animated": True,
            "style": {"stroke": "#48bb78", "strokeWidth": 2},
            "label": "Achieve Goal"
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "totalStages": len(stages),
            "totalEvents": sum(len(s.get("event_ids", [])) for s in stages),
            "userProfile": profile
        }
    }


# =========================
# PUBLIC API
# =========================

def build_career_graph_for_user(user_id: str) -> Dict:
    """
    Main entrypoint: Create complete career roadmap
    
    1. RAG retrieval of relevant events
    2. LLM generates roadmap structure
    3. Convert to React Flow with enhanced visualization
    """
    # Get events and profile
    events, profile = rag_retrieve_events_for_user(
        user_id=user_id,
        k=25,
        min_sim=0.15,
    )

    if not events:
        return {
            "nodes": [],
            "edges": [],
            "metadata": {"error": "No relevant events found"}
        }

    # Create lookup for event details
    events_lookup = {ev["event_id"]: ev for ev in events}

    # Generate roadmap structure
    roadmap = call_llm_for_graph(profile, events)
    
    # Convert to React Flow format
    graph = roadmap_to_react_flow(roadmap, profile, events_lookup)
    
    return graph


if __name__ == "__main__":
    # Test with a user ID
    test_user_id = os.getenv("TEST_USER_ID", "c002d160a44d")
    graph = build_career_graph_for_user(test_user_id)

    # Save to a JSON file
    output_path = "career_graph.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print(f"Graph saved to {output_path}")