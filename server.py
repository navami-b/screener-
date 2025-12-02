from fastapi import FastAPI, APIRouter, HTTPException, Depends, status # pyright: ignore[reportMissingImports]
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # pyright: ignore[reportMissingImports]
from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
from starlette.middleware.cors import CORSMiddleware # pyright: ignore[reportMissingImports]
from motor.motor_asyncio import AsyncIOMotorClient # pyright: ignore[reportMissingImports]
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timezone, timedelta
import bcrypt # pyright: ignore[reportMissingImports]
import jwt # pyright: ignore[reportMissingImports]
from emergentintegrations.llm.chat import LlmChat, UserMessage # pyright: ignore[reportMissingImports]

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'screener_jwt_secret_key_2025')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

# Emergent LLM Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

security = HTTPBearer()

# ================ MODELS ================

class UserRegister(BaseModel):
    email: str
    password: str
    name: str
    role: str  # 'admin' or 'candidate'

class UserLogin(BaseModel):
    email: str
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    role: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DriveCreate(BaseModel):
    title: str
    topic: str
    num_questions: int
    description: Optional[str] = ""

class Drive(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    topic: str
    num_questions: int
    description: str
    drive_code: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    admin_id: str
    questions_generated: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Question(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drive_id: str
    question_text: str
    question_number: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class TestSessionCreate(BaseModel):
    drive_code: str
    candidate_name: str
    candidate_email: str

class TestSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    drive_id: str
    candidate_name: str
    candidate_email: str
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    status: str = "in_progress"  # in_progress, completed, evaluated

class AnswerSubmit(BaseModel):
    session_id: str
    question_id: str
    answer_text: str

class Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    question_id: str
    answer_text: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    submitted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Result(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    drive_id: str
    total_score: float
    max_score: float
    percentage: float
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ================ AUTH HELPERS ================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def create_jwt_token(user_id: str, email: str, role: str) -> str:
    expiration = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        'user_id': user_id,
        'email': email,
        'role': role,
        'exp': expiration
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_jwt_token(token)
    user = await db.users.find_one({"id": payload['user_id']}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return User(**user)

async def get_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# ================ AUTH ROUTES ================

@api_router.post("/auth/register")
async def register(user_data: UserRegister):
    # Check if user exists
    existing_user = await db.users.find_one({"email": user_data.email}, {"_id": 0})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=user_data.email,
        name=user_data.name,
        role=user_data.role
    )
    user_dict = user.model_dump()
    user_dict['password'] = hash_password(user_data.password)
    
    await db.users.insert_one(user_dict)
    
    # Generate token
    token = create_jwt_token(user.id, user.email, user.role)
    
    return {
        "message": "User registered successfully",
        "token": token,
        "user": user.model_dump()
    }

@api_router.post("/auth/login")
async def login(login_data: UserLogin):
    # Find user
    user_dict = await db.users.find_one({"email": login_data.email}, {"_id": 0})
    if not user_dict:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not verify_password(login_data.password, user_dict['password']):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate token
    token = create_jwt_token(user_dict['id'], user_dict['email'], user_dict['role'])
    
    user = User(**user_dict)
    
    return {
        "message": "Login successful",
        "token": token,
        "user": user.model_dump()
    }

@api_router.get("/auth/me")
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# ================ ADMIN ROUTES ================

@api_router.post("/admin/drives", response_model=Drive)
async def create_drive(drive_data: DriveCreate, current_user: User = Depends(get_admin_user)):
    drive = Drive(
        title=drive_data.title,
        topic=drive_data.topic,
        num_questions=drive_data.num_questions,
        description=drive_data.description,
        admin_id=current_user.id
    )
    
    drive_dict = drive.model_dump()
    await db.drives.insert_one(drive_dict)
    
    return drive

@api_router.get("/admin/drives", response_model=List[Drive])
async def get_drives(current_user: User = Depends(get_admin_user)):
    drives = await db.drives.find({"admin_id": current_user.id}, {"_id": 0}).to_list(100)
    return drives

@api_router.get("/admin/drives/{drive_id}", response_model=Drive)
async def get_drive(drive_id: str, current_user: User = Depends(get_admin_user)):
    drive = await db.drives.find_one({"id": drive_id, "admin_id": current_user.id}, {"_id": 0})
    if not drive:
        raise HTTPException(status_code=404, detail="Drive not found")
    return Drive(**drive)

@api_router.post("/admin/drives/{drive_id}/generate-questions")
async def generate_questions(drive_id: str, current_user: User = Depends(get_admin_user)):
    # Get drive
    drive = await db.drives.find_one({"id": drive_id, "admin_id": current_user.id}, {"_id": 0})
    if not drive:
        raise HTTPException(status_code=404, detail="Drive not found")
    
    # Check if questions already generated
    if drive.get('questions_generated'):
        raise HTTPException(status_code=400, detail="Questions already generated for this drive")
    
    # Initialize AI chat
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"drive_{drive_id}",
        system_message="You are an expert Python programming instructor who creates challenging and educational coding questions."
    )
    chat.with_model("openai", "gpt-4o-mini")
    
    # Generate questions
    prompt = f"""Generate {drive['num_questions']} Python coding questions on the topic: {drive['topic']}.
    
Each question should:
1. Be clear and specific
2. Test practical programming skills
3. Be answerable with code snippets or explanations
4. Cover different aspects of {drive['topic']}

Format: Return ONLY the questions, one per line, numbered 1., 2., 3., etc. Do not include any other text."""
    
    user_message = UserMessage(text=prompt)
    response = await chat.send_message(user_message)
    
    # Parse questions
    questions_text = response.strip().split('\n')
    questions = []
    question_number = 1
    
    for line in questions_text:
        line = line.strip()
        if line and len(line) > 10:  # Filter out empty or very short lines
            # Remove numbering if present
            cleaned_line = line
            for i in range(1, 100):
                if line.startswith(f"{i}.") or line.startswith(f"{i})"):
                    cleaned_line = line[len(str(i))+1:].strip()
                    break
            
            if cleaned_line:
                question = Question(
                    drive_id=drive_id,
                    question_text=cleaned_line,
                    question_number=question_number
                )
                questions.append(question.model_dump())
                question_number += 1
                
                if question_number > drive['num_questions']:
                    break
    
    # Store questions
    if questions:
        await db.questions.insert_many(questions)
        await db.drives.update_one(
            {"id": drive_id},
            {"$set": {"questions_generated": True}}
        )
    
    return {"message": f"Generated {len(questions)} questions successfully", "count": len(questions)}

@api_router.get("/admin/drives/{drive_id}/results")
async def get_drive_results(drive_id: str, current_user: User = Depends(get_admin_user)):
    # Verify drive belongs to admin
    drive = await db.drives.find_one({"id": drive_id, "admin_id": current_user.id}, {"_id": 0})
    if not drive:
        raise HTTPException(status_code=404, detail="Drive not found")
    
    # Get all sessions for this drive
    sessions = await db.test_sessions.find({"drive_id": drive_id}, {"_id": 0}).to_list(100)
    
    results_data = []
    for session in sessions:
        result = await db.results.find_one({"session_id": session['id']}, {"_id": 0})
        if result:
            results_data.append({
                "session": session,
                "result": result
            })
    
    return results_data

# ================ CANDIDATE ROUTES ================

@api_router.get("/candidate/drive/{drive_code}")
async def get_drive_by_code(drive_code: str):
    drive = await db.drives.find_one({"drive_code": drive_code.upper()}, {"_id": 0})
    if not drive:
        raise HTTPException(status_code=404, detail="Invalid drive code")
    
    # Return drive info without questions
    return {
        "id": drive['id'],
        "title": drive['title'],
        "topic": drive['topic'],
        "description": drive['description'],
        "num_questions": drive['num_questions']
    }

@api_router.post("/candidate/start-test", response_model=TestSession)
async def start_test(session_data: TestSessionCreate):
    # Verify drive exists
    drive = await db.drives.find_one({"drive_code": session_data.drive_code.upper()}, {"_id": 0})
    if not drive:
        raise HTTPException(status_code=404, detail="Invalid drive code")
    
    if not drive.get('questions_generated'):
        raise HTTPException(status_code=400, detail="Test questions are not ready yet")
    
    # Create test session
    session = TestSession(
        drive_id=drive['id'],
        candidate_name=session_data.candidate_name,
        candidate_email=session_data.candidate_email
    )
    
    session_dict = session.model_dump()
    await db.test_sessions.insert_one(session_dict)
    
    return session

@api_router.get("/candidate/questions/{session_id}", response_model=List[Question])
async def get_questions_for_session(session_id: str):
    # Get session
    session = await db.test_sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get questions for this drive
    questions = await db.questions.find({"drive_id": session['drive_id']}, {"_id": 0}).sort("question_number", 1).to_list(100)
    
    return questions

@api_router.post("/candidate/submit-answer", response_model=Answer)
async def submit_answer(answer_data: AnswerSubmit):
    # Verify session exists
    session = await db.test_sessions.find_one({"id": answer_data.session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session['status'] != 'in_progress':
        raise HTTPException(status_code=400, detail="Test is already completed")
    
    # Check if answer already exists
    existing_answer = await db.answers.find_one({
        "session_id": answer_data.session_id,
        "question_id": answer_data.question_id
    }, {"_id": 0})
    
    if existing_answer:
        # Update existing answer
        await db.answers.update_one(
            {"id": existing_answer['id']},
            {"$set": {"answer_text": answer_data.answer_text}}
        )
        existing_answer['answer_text'] = answer_data.answer_text
        return Answer(**existing_answer)
    else:
        # Create new answer
        answer = Answer(
            session_id=answer_data.session_id,
            question_id=answer_data.question_id,
            answer_text=answer_data.answer_text
        )
        
        answer_dict = answer.model_dump()
        await db.answers.insert_one(answer_dict)
        
        return answer

@api_router.post("/candidate/complete-test/{session_id}")
async def complete_test(session_id: str):
    # Get session
    session = await db.test_sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session['status'] != 'in_progress':
        raise HTTPException(status_code=400, detail="Test is already completed")
    
    # Get all answers for this session
    answers = await db.answers.find({"session_id": session_id}, {"_id": 0}).to_list(100)
    
    # Get questions
    questions = await db.questions.find({"drive_id": session['drive_id']}, {"_id": 0}).to_list(100)
    
    # Initialize AI for evaluation
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=f"eval_{session_id}",
        system_message="You are an expert Python programming evaluator. Evaluate answers objectively and provide constructive feedback."
    )
    chat.with_model("openai", "gpt-4o-mini")
    
    total_score = 0.0
    max_score = len(questions) * 10.0  # 10 points per question
    
    # Evaluate each answer
    for answer in answers:
        # Find corresponding question
        question = next((q for q in questions if q['id'] == answer['question_id']), None)
        if not question:
            continue
        
        # Evaluate answer
        eval_prompt = f"""Question: {question['question_text']}

Candidate's Answer: {answer['answer_text']}

Evaluate this answer on a scale of 0-10 based on:
1. Correctness
2. Completeness
3. Code quality (if applicable)
4. Understanding of concepts

Provide:
1. A score (0-10)
2. Brief feedback (2-3 sentences)

Format your response as:
Score: [number]
Feedback: [your feedback]"""
        
        user_message = UserMessage(text=eval_prompt)
        evaluation = await chat.send_message(user_message)
        
        # Parse evaluation
        score = 5.0  # Default score
        feedback = evaluation
        
        try:
            lines = evaluation.strip().split('\n')
            for line in lines:
                if line.startswith('Score:'):
                    score_text = line.replace('Score:', '').strip()
                    score = float(score_text)
                elif line.startswith('Feedback:'):
                    feedback = line.replace('Feedback:', '').strip()
        except:
            pass
        
        # Update answer with score and feedback
        await db.answers.update_one(
            {"id": answer['id']},
            {"$set": {"score": score, "feedback": feedback}}
        )
        
        total_score += score
    
    # Calculate percentage
    percentage = (total_score / max_score * 100) if max_score > 0 else 0
    
    # Create result
    result = Result(
        session_id=session_id,
        drive_id=session['drive_id'],
        total_score=total_score,
        max_score=max_score,
        percentage=percentage
    )
    
    result_dict = result.model_dump()
    await db.results.insert_one(result_dict)
    
    # Update session status
    await db.test_sessions.update_one(
        {"id": session_id},
        {"$set": {
            "status": "evaluated",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {
        "message": "Test completed and evaluated successfully",
        "result": result_dict
    }

@api_router.get("/candidate/results/{session_id}")
async def get_results(session_id: str):
    # Get session
    session = await db.test_sessions.find_one({"id": session_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get result
    result = await db.results.find_one({"session_id": session_id}, {"_id": 0})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Get all answers with feedback
    answers = await db.answers.find({"session_id": session_id}, {"_id": 0}).to_list(100)
    
    # Get questions
    questions = await db.questions.find({"drive_id": session['drive_id']}, {"_id": 0}).to_list(100)
    
    # Combine data
    detailed_results = []
    for answer in answers:
        question = next((q for q in questions if q['id'] == answer['question_id']), None)
        if question:
            detailed_results.append({
                "question": question['question_text'],
                "answer": answer['answer_text'],
                "score": answer.get('score', 0),
                "feedback": answer.get('feedback', '')
            })
    
    return {
        "session": session,
        "result": result,
        "detailed_results": detailed_results
    }

# ================ ROOT ROUTE ================

@api_router.get("/")
async def root():
    return {"message": "Screener API is running"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
