from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager, contextmanager
import os
import uuid

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

import pymysql
from pymysql.cursors import DictCursor
import json
import numpy as np
from pydantic import BaseModel

from career_graph import build_career_graph_for_user

from dotenv import load_dotenv
load_dotenv()
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# ---------- Database Helpers ----------

@contextmanager
def get_db_connection():
    """Context manager for MySQL database connections"""
    conn = None
    try:
        conn = pymysql.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=int(os.getenv("DB_PORT", "3306")),
            cursorclass=DictCursor
        )
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def parse_json_field(value):
    """Convert JSON columns from strings/bytes to native Python objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value

def create_users_table():
    """Create users table if it doesn't exist"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    year VARCHAR(50) NOT NULL,
                    faculty VARCHAR(255) NOT NULL,
                    interests TEXT NOT NULL,
                    end_goal TEXT NOT NULL,
                    timeline VARCHAR(255),
                    resume_key VARCHAR(500),
                    embedding JSON,
                    career_path JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)

def insert_user(user_data: dict):
    """Insert a new user into the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Convert embedding list to JSON string if it exists
            if 'embedding' in user_data and user_data['embedding']:
                user_data['embedding'] = json.dumps(user_data['embedding'])
            
            cur.execute("""
                INSERT INTO users (
                    user_id, name, email, year, faculty, 
                    interests, end_goal, timeline, resume_key, embedding
                ) VALUES (
                    %(user_id)s, %(name)s, %(email)s, %(year)s, %(faculty)s,
                    %(interests)s, %(end_goal)s, %(timeline)s, %(resume_key)s, %(embedding)s
                )
            """, user_data)
            return user_data['user_id']

def get_user_by_email(email: str):
    """Check if user exists by email"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            return cur.fetchone()

def get_user_by_id(user_id: str):
    """Get user by ID"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cur.fetchone()
            # Parse JSON columns back to Python objects
            if user:
                embedding = parse_json_field(user.get('embedding'))
                career_path = parse_json_field(user.get('career_path'))
                if embedding is not None:
                    user['embedding'] = embedding
                if career_path is not None:
                    user['career_path'] = career_path
            return user
        
def update_user_career_path(user_id: str, career_path: Dict[str, Any]):
    """Persist an updated career path for the user."""
    if career_path is None:
        raise ValueError("career_path cannot be None")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET career_path = %s WHERE user_id = %s",
                (json.dumps(career_path), user_id)
            )
            if cur.rowcount == 0:
                raise ValueError(f"User {user_id} not found")

def get_all_users_with_embeddings():
    """Get all users with their embeddings for similarity search"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, name, email, faculty, interests, end_goal, embedding FROM users WHERE embedding IS NOT NULL")
            users = cur.fetchall()
            # Parse embedding JSON for each user
            for user in users:
                if user.get('embedding'):
                    embedding = parse_json_field(user['embedding'])
                    if embedding is not None:
                        user['embedding'] = embedding
            return users

# ---------- Embedding Helpers ----------

def get_bedrock_client():
    """Initialize Bedrock Runtime client"""
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv("AWS_REGION", "us-west-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

def create_user_embedding(user_data: dict) -> list:
    """
    Create an embedding from user data using Amazon Titan Embeddings
    Titan Text Embeddings V2 produces 1024-dimensional embeddings
    """
    # Combine user information into a single text
    text_to_embed = f"""
    Name: {user_data['name']}
    Faculty: {user_data['faculty']}
    Year: {user_data['year']}
    Interests: {user_data['interests']}
    Goal: {user_data['end_goal']}
    Timeline: {user_data.get('timeline', 'Not specified')}
    """.strip()
    
    try:
        bedrock = get_bedrock_client()
        
        # Prepare request body for Titan Embeddings V2
        request_body = json.dumps({
            "inputText": text_to_embed
        })
        
        # Invoke the model
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",  # Titan Text Embeddings V2 (1024 dimensions)
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        embedding = response_body['embedding']
        
        return embedding
        
    except Exception as e:
        print(f"✗ Failed to create embedding: {str(e)}")
        raise

def cosine_similarity(embedding1: list, embedding2: list) -> float:
    """Calculate cosine similarity between two embeddings"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ---------- Career Path Chat Helpers ----------

CHAT_SYSTEM_PROMPT = """
You are PathFinder UBC, an AI assistant that edits existing student career roadmaps.

You will be given JSON with:
- student_profile: name, faculty, year, interests, end_goal, timeline.
- current_career_path: a React Flow compatible graph object with nodes, edges, and metadata.
- user_message: natural language guidance from the student describing desired changes.

Update the roadmap so it reflects the user's latest intent.

Rules:
1. Always respond with valid JSON (no markdown) containing the keys: nodes, edges, metadata.
2. Preserve React Flow structure and reuse existing node/edge IDs whenever possible. Only add new IDs if needed and keep them descriptive.
3. Do not invent new event_ids that are not already present in the current_career_path. Reorder, relabel, or reprioritize existing events/stages instead.
4. Ensure metadata stays consistent (update counts or add an "explanation" note when changes are made).
5. If a request cannot be satisfied exactly, make the closest reasonable adjustment and explain the limitation inside metadata.explanation.
""".strip()

def build_chat_update_payload(user: Dict[str, Any], message: str) -> str:
    """Serialize the data the LLM needs to understand the requested change."""
    payload = {
        "student_profile": {
            "name": user.get("name"),
            "faculty": user.get("faculty"),
            "year": user.get("year"),
            "interests": user.get("interests"),
            "end_goal": user.get("end_goal"),
            "timeline": user.get("timeline")
        },
        "current_career_path": user.get("career_path"),
        "user_message": message
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def extract_bedrock_text_response(resp_body: Dict[str, Any]) -> str:
    """Handle different Bedrock response payload formats and return the text."""
    if "content" in resp_body:
        text = ""
        for block in resp_body["content"]:
            if block.get("type") == "text":
                text += block.get("text", "")
        return text
    if "outputText" in resp_body:
        return resp_body["outputText"]
    raise RuntimeError(f"Unexpected Bedrock response format: {resp_body}")

def update_career_path_with_message(user: Dict[str, Any], message: str) -> Dict[str, Any]:
    """
    Use the existing career_path plus a free-form message to build a refreshed graph.
    """
    career_path = user.get("career_path")
    if not career_path:
        raise ValueError("User has no stored career_path to update")
    
    bedrock = get_bedrock_client()
    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "temperature": 0.2,
            "system": CHAT_SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": build_chat_update_payload(user, message)
                }
            ]
        }
    )
    
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=request_body,
        contentType="application/json",
        accept="application/json"
    )
    
    resp_body = json.loads(response["body"].read())
    text = extract_bedrock_text_response(resp_body)
    
    try:
        updated_path = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {text}") from exc
    
    return updated_path

# ---------- Lifespan Event Handler ----------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        create_users_table()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"✗ Database initialization failed: {str(e)}")
    
    yield
    
    # Shutdown
    print("✓ Application shutdown")

app = FastAPI(lifespan=lifespan)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- S3 Helpers ----------

def get_s3_client():
    """Initialize and return S3 client with credentials from environment"""
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )

def upload_to_s3(file_content: bytes, filename: str, content_type: str, user_id: str) -> str:
    """Upload file to S3 and return the object key"""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable not set")
    
    # Create a unique key for the file
    file_extension = filename.split('.')[-1] if '.' in filename else 'pdf'
    key = f"{user_id}.{file_extension}"
    
    try:
        s3 = get_s3_client()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=file_content,
            ContentType=content_type,
            Metadata={
                'original_filename': filename,
                'user_id': user_id
            }
        )
        return key
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        raise Exception(f"Failed to upload to S3: {str(e)}")

# ---------- Routes ----------

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/register")
async def register_user(
    name: str = Form(...),
    email: str = Form(...),
    year: str = Form(...),
    faculty: str = Form(...),
    interests: str = Form(...),
    end_goal: str = Form(...),
    timeline: Optional[str] = Form(""),
    resume: Optional[UploadFile] = File(None)
):
    """
    Single endpoint to register user with all data including optional resume upload.
    Saves user to RDS MySQL database and uploads resume to S3.
    """
    
    # Check if user already exists
    try:
        existing_user = get_user_by_email(email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    
    # Generate a user ID
    user_id = uuid.uuid4().hex[:12]
    
    resume_key = None
    
    # Handle resume upload if provided
    if resume and resume.filename:
        # Validate file type
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]
        
        if resume.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, DOC, and DOCX files are allowed."
            )
        
        # Validate file size (max 10MB)
        file_content = await resume.read()
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 10MB limit."
            )
        
        try:
            # Upload to S3
            resume_key = upload_to_s3(
                file_content=file_content,
                filename=resume.filename,
                content_type=resume.content_type,
                user_id=user_id
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload resume: {str(e)}"
            )
    
    # Prepare user data for database
    user_data = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "year": year,
        "faculty": faculty,
        "interests": interests,
        "end_goal": end_goal,
        "timeline": timeline if timeline else None,
        "resume_key": resume_key,
        "embedding": None
    }
    
    # Create embedding from user data
    try:
        embedding = create_user_embedding(user_data)
        user_data["embedding"] = embedding
        print(f"✓ Created embedding with {len(embedding)} dimensions")
    except Exception as e:
        print(f"⚠ Warning: Failed to create embedding: {str(e)}")
        # Continue without embedding - don't fail the registration
    
    # Save to database
    try:
        inserted_user_id = insert_user(user_data)
        if not inserted_user_id:
            raise Exception("Failed to insert user")
        
        print(f"✓ User registered: {user_id}")
        
        return {
            "message": "User registered successfully!",
            "user_id": user_id,
            "resume_uploaded": resume_key is not None,
            "resume_key": resume_key,
            "embedding_created": user_data["embedding"] is not None
        }
    except Exception as e:
        print(f"✗ Failed to save user to database: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save user: {str(e)}"
        )

@app.get("/health")
def health_check():
    """Health check endpoint - checks database connection"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }

@app.get("/user/{user_id}")
def get_user(user_id: str):
    """Get user details by user_id"""
    try:
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

@app.get("/resume/download-url/{user_id}")
def get_resume_download_url(user_id: str, expires_in: int = 3600):
    """Generate a pre-signed URL to download a user's resume"""
    
    # Get user from database to find resume_key
    try:
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.get('resume_key'):
            raise HTTPException(status_code=404, detail="User has no resume")
        
        resume_key = user['resume_key']
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )
    
    # Generate pre-signed URL
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise HTTPException(status_code=500, detail="S3_BUCKET not configured")
    
    try:
        s3 = get_s3_client()
        download_url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': bucket, 'Key': resume_key},
            ExpiresIn=expires_in
        )
        return {"download_url": download_url}
    except (BotoCoreError, NoCredentialsError, ClientError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate download URL: {str(e)}"
        )

@app.post("/search/similar-users")
async def search_similar_users(
    query: str = Form(...),
    top_k: int = Form(5)
):
    """
    Find similar users based on a text query using embeddings
    Returns the top_k most similar users
    """
    try:
        # Create embedding for the search query
        query_data = {"name": "", "faculty": "", "year": "", "interests": query, "end_goal": query}
        query_embedding = create_user_embedding(query_data)
        
        # Get all users with embeddings
        users = get_all_users_with_embeddings()
        
        if not users:
            return {
                "message": "No users with embeddings found",
                "results": []
            }
        
        # Calculate similarity scores
        results = []
        for user in users:
            if user.get('embedding'):
                similarity = cosine_similarity(query_embedding, user['embedding'])
                results.append({
                    "user_id": user['user_id'],
                    "name": user['name'],
                    "email": user['email'],
                    "faculty": user['faculty'],
                    "interests": user['interests'],
                    "end_goal": user['end_goal'],
                    "similarity_score": float(similarity)
                })
        
        # Sort by similarity (highest first) and get top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_results = results[:top_k]
        
        return {
            "query": query,
            "total_users_searched": len(users),
            "results": top_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """Receive a chat message and refresh the stored career path for a user."""
    cleaned_message = payload.message.strip()
    if not cleaned_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        user = get_user_by_id(payload.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    try:
        updated_career_path = update_career_path_with_message(user, cleaned_message)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update career path: {str(e)}")
    
    try:
        update_user_career_path(payload.user_id, updated_career_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save career path: {str(e)}")
    
    return {
        "message": "Career path updated",
        "career_path": updated_career_path
    }
        
@app.get("/users/{user_id}/career-graph")
def get_career_graph(user_id: str):
    return build_career_graph_for_user(user_id)
