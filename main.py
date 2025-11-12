from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class User(BaseModel):
    email: str
    year: str
    name: str
    faculty: str
    interests: Optional[str] = None
    end_goal: Optional[str] = None
    timeline: Optional[str] = None

@app.post("/register")
async def register_user(
    email: str = Form(...),
    year: str = Form(...),
    name: str = Form(...),
    faculty: str = Form(...),
    interests: Optional[str] = Form(None),
    end_goal: Optional[str] = Form(None),
    timeline: Optional[str] = Form(None),
    resume: Optional[UploadFile] = File(None)
):
    data = {
        "email": email,
        "year": year,
        "name": name,
        "faculty": faculty,
        "interests": interests,
        "end_goal": end_goal,
        "timeline": timeline,
        "resume_filename": resume.filename if resume else None
    }
    return {"message": "User registered successfully!", "data": data}
