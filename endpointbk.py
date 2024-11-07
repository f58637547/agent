from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from runnable import create_runnable  # Imports LangGraph runnable setup

load_dotenv()

app = FastAPI(
    title="GROQ-powered AI Chat",
    version="1.0",
    description="LangGraph backend with GROQ integration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

agent = create_runnable()

class ChatRequest(BaseModel):
    question: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    state = {"messages": [{"user": request.question, "session_id": request.session_id}]}
    try:
        response = await agent.invoke(state)
        return {"response": response["messages"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the GROQ-powered AI chat!"}