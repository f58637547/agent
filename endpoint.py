from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from runnable import create_runnable  # Imports LangGraph runnable setup
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitSDK, LangGraphAgent

load_dotenv()

app = FastAPI(
    title="GROQ-powered AI Chat",
    version="1.0",
    description="LangGraph backend with GROQ integration.",
)

# Add HTTPS redirect middleware to ensure FastAPI treats all requests as HTTPS
app.add_middleware(HTTPSRedirectMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize your LangGraph agent
agent = create_runnable()

# Set up CopilotKit SDK with LangGraph agent
sdk = CopilotKitSDK(
    agents=[
        LangGraphAgent(
            name="search_agent",
            description="GROQ-powered search agent",
            agent=agent,
        )
    ],
)

# Register the /copilotkit endpoint for CopilotKit requests
add_fastapi_endpoint(app, sdk, "/copilotkit")

# Define the existing /chat endpoint
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

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the GROQ-powered AI chat!"}
