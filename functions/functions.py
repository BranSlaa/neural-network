from typing import List, Optional
from fastapi import HTTPException, Security
from classes.classes import SubscriptionTier, TokenData, User, Event
import jwt
import hashlib
import json
import string
from nltk.stem import WordNetLemmatizer
from contextlib import contextmanager
from psycopg2.pool import SimpleConnectionPool
import os
from nltk.corpus import stopwords
from psycopg2.extras import RealDictCursor, Json
from openai import OpenAI
from fastapi.security import APIKeyHeader

global index

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:brandons-local-server@localhost:5432/history_map_db"
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

# Create a connection pool
conn_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

# Context manager for database connections
@contextmanager
def get_db_connection():
    conn = conn_pool.getconn()
    try:
        conn.autocommit = True
        yield conn
    finally:
        conn_pool.putconn(conn)

# Context manager for database cursors
@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
        finally:
            cursor.close()
            
def get_current_user_tier(authorization: str = Security(api_key_header)) -> TokenData:
    """
    Validate the JWT token and return the user's tier
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split("Bearer ")[1]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        clerk_id = payload.get("clerk_id")
        tier = payload.get("tier")
        
        if not clerk_id or not tier:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        
        return TokenData(clerk_id=clerk_id, tier=tier)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_tier_access(required_tier: SubscriptionTier, user_tier: SubscriptionTier) -> bool:
    """
    Check if the user's tier meets the required tier level
    Tier hierarchy: historian > scholar > student
    """
    tier_levels = {
        SubscriptionTier.STUDENT: 1,
        SubscriptionTier.SCHOLAR: 2,
        SubscriptionTier.HISTORIAN: 3
    }
    
    return tier_levels[user_tier] >= tier_levels[required_tier]

async def get_user_from_db(clerk_id: str) -> Optional[User]:
    """
    Retrieve user information from the database
    """
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT id, clerk_id, username, email, subscription_tier, created_at
            FROM users
            WHERE clerk_id = %s
            """,
            (clerk_id,)
        )
        user_data = cursor.fetchone()
        
        if user_data:
            return User(
                id=str(user_data['id']),
                clerk_id=user_data['clerk_id'],
                username=user_data['username'],
                email=user_data['email'],
                subscription_tier=user_data['subscription_tier'],
                created_at=str(user_data['created_at'])
            )
        return None

# Helper functions
def get_cache_key(topic, title, yearMin, yearMax, subjects):
    key_data = {"topic": topic, "title": title, "yearMin": yearMin, "yearMax": yearMax, "subjects": subjects}
    return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()

lemmatizer = WordNetLemmatizer()
def extract_key_terms(text: str) -> List[str]:
    translator = str.maketrans('', '', string.punctuation)
    cleaned = text.translate(translator).lower()
    words = cleaned.split()
    stopword_set = set(stopwords)
    keywords = [lemmatizer.lemmatize(w, pos='n') for w in words if w not in stopword_set]
    return list(set(keywords))

def normalize_term(term: str) -> str:
    return lemmatizer.lemmatize(term.lower(), pos='n')

def get_event_embedding(event: Event) -> List[float]:
    input_text = f"Title: {event.title}. Year: {event.year}. Location: {event.lat}, {event.lon}. Subject: {event.subject}. Info: {event.info}"
    embedding_response = client.embeddings.create(input=input_text, model="text-embedding-ada-002")
    return embedding_response.data[0].embedding

# Store event in database and Pinecone
def store_event(event: Event):
    # Store in database
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO events (id, title, year, lat, lon, subject, info, key_terms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (event.id, event.title, float(event.year), event.lat, event.lon, 
            event.subject, event.info, Json(event.key_terms if event.key_terms else []))
        )
    
    # Store in Pinecone
    try:
        key_terms = extract_key_terms(event.title + " " + event.info)
        metadata = {
            "title": event.title,
            "year": float(event.year),
            "lat": event.lat,
            "lon": event.lon,
            "subject": event.subject,
            "info": event.info
        }
        
        # Add each term to metadata
        for term in key_terms:
            metadata[normalize_term(term)] = True
            
        vector = get_event_embedding(event)
        record = {
            "id": event.id,
            "values": vector,
            "metadata": metadata
        }
        index.upsert(vectors=[record])
        print(f"Stored event in Pinecone: {event.id}")
    except Exception as e:
        print(f"Error storing in Pinecone: {e}")