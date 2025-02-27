import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import nltk
from functions.functions import *
from functions.database import init_db

# Import endpoint modules
from endpoints import auth, users, misc, get_events

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize clients and configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "history-map")
pinecone_dim = int(os.getenv("PINECONE_DIMENSION", "1536"))
pc = Pinecone(api_key=pinecone_api_key)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # List specific origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = None
@app.on_event("startup")
async def startup_event():
    global index
    # Initialize Pinecone index
    try:
        if pinecone_index_name not in pc.list_indexes().names():
            pc.create_index(
                name=pinecone_index_name,
                dimension=pinecone_dim,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
    except Exception as e:
        print(f"Error during index creation: {e}")
    
    index = pc.Index(pinecone_index_name)
    
    # Initialize database
    init_db()
    
    # Make index available to get_events module
    get_events.index = index

# Set up routes from each endpoint module
auth.setup_routes(app)
users.setup_routes(app)
misc.setup_routes(app)
get_events.setup_routes(app)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

