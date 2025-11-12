from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from contextlib import asynccontextmanager, contextmanager
import os
import uuid

import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

import pymysql
from pymysql.cursors import DictCursor

from dotenv import load_dotenv
load_dotenv()

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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)

def insert_user(user_data: dict):
    """Insert a new user into the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (
                    user_id, name, email, year, faculty, 
                    interests, end_goal, timeline, resume_key
                ) VALUES (
                    %(user_id)s, %(name)s, %(email)s, %(year)s, %(faculty)s,
                    %(interests)s, %(end_goal)s, %(timeline)s, %(resume_key)s
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
            return cur.fetchone()

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
        "resume_key": resume_key
    }
    
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
            "resume_key": resume_key
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