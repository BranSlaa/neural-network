import os
import json
import re
import string
import hashlib
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, Depends, Header, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from enum import Enum
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import nltk
from nltk.stem import WordNetLemmatizer
from stop_words import stopwords
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
import jwt

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize clients and configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "history-map")
pinecone_dim = int(os.getenv("PINECONE_DIMENSION", "1536"))
pc = Pinecone(api_key=pinecone_api_key)

# JWT settings
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # List specific origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Subscription tier enum
class SubscriptionTier(str, Enum):
    STUDENT = "student"
    SCHOLAR = "scholar"
    HISTORIAN = "historian"

# Pydantic models
class Event(BaseModel):
    id: Optional[str] = None
    title: str
    year: int
    lat: float
    lon: float
    subject: str
    info: str
    key_terms: List[str] = []

class User(BaseModel):
    id: str
    clerk_id: str
    username: str
    email: str
    subscription_tier: SubscriptionTier
    created_at: Optional[str] = None

class UserCreate(BaseModel):
    clerk_id: str
    username: str
    email: str
    subscription_tier: SubscriptionTier = SubscriptionTier.STUDENT

class TokenData(BaseModel):
    clerk_id: str
    tier: SubscriptionTier

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:brandons-local-server@localhost:5432/history_map_db"

# Create a connection pool
conn_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=DATABASE_URL
)

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

# Initialize database tables
def init_db():
    tables = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            clerk_id VARCHAR NOT NULL UNIQUE,
            username VARCHAR NOT NULL,
            email VARCHAR NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            subscription_tier VARCHAR NOT NULL DEFAULT 'student' CHECK (subscription_tier IN ('student', 'scholar', 'historian'))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id VARCHAR PRIMARY KEY,
            title VARCHAR NOT NULL,
            year FLOAT NOT NULL,
            lat FLOAT NOT NULL,
            lon FLOAT NOT NULL,
            subject VARCHAR NOT NULL,
            info TEXT NOT NULL,
            key_terms JSONB
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS quizzes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title VARCHAR NOT NULL,
            description TEXT,
            difficulty VARCHAR CHECK (difficulty IN ('easy', 'medium', 'hard')),
            required_tier VARCHAR NOT NULL CHECK (required_tier IN ('student', 'scholar', 'historian')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_quiz_attempts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            quiz_id UUID NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
            score INTEGER NOT NULL,
            attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS leaderboard (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
            total_score INTEGER NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS vectorized_content (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            content TEXT NOT NULL,
            vector_embedding JSONB,
            metadata_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS event_user_vectors (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            vector_id UUID NOT NULL REFERENCES vectorized_content(id) ON DELETE CASCADE,
            interaction_type VARCHAR CHECK (interaction_type IN ('viewed', 'favorited', 'searched')),
            interaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS trade_routes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR NOT NULL,
            historical_period VARCHAR NOT NULL,
            origin VARCHAR NOT NULL,
            destination VARCHAR NOT NULL,
            goods JSONB,
            required_tier VARCHAR NOT NULL CHECK (required_tier IN ('student', 'scholar', 'historian')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_trades (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            route_id UUID NOT NULL REFERENCES trade_routes(id) ON DELETE CASCADE,
            goods_traded JSONB,
            profit INTEGER,
            trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    with get_db_cursor() as cursor:
        for table in tables:
            cursor.execute(table)

# Authentication and authorization
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

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

# User management endpoints
@app.post("/users", response_model=User)
async def create_user(user: UserCreate):
    """
    Create a new user with the specified subscription tier
    """
    with get_db_cursor() as cursor:
        try:
            # Print the request for debugging
            print(f"Creating user: {user}")
            
            # Ensure subscription tier is valid
            tier = SubscriptionTier.STUDENT
            if hasattr(user, 'subscription_tier') and user.subscription_tier:
                if user.subscription_tier in [t.value for t in SubscriptionTier]:
                    tier = user.subscription_tier
                else:
                    print(f"Invalid tier provided: {user.subscription_tier}, using default: student")
            
            cursor.execute(
                """
                INSERT INTO users (clerk_id, username, email, subscription_tier)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (clerk_id) DO UPDATE SET
                    username = EXCLUDED.username,
                    email = EXCLUDED.email,
                    subscription_tier = EXCLUDED.subscription_tier
                RETURNING id, clerk_id, username, email, subscription_tier, created_at
                """,
                (user.clerk_id, user.username, user.email, tier)
            )
            new_user = cursor.fetchone()
            return User(
                id=str(new_user['id']),
                clerk_id=new_user['clerk_id'],
                username=new_user['username'],
                email=new_user['email'],
                subscription_tier=new_user['subscription_tier'],
                created_at=str(new_user['created_at'])
            )
        except Exception as e:
            print(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.get("/users/me", response_model=User)
async def get_current_user(token_data: TokenData = Depends(get_current_user_tier)):
    """
    Get the current authenticated user
    """
    user = await get_user_from_db(token_data.clerk_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.put("/users/me/tier", response_model=User)
async def update_user_tier(
    new_tier: SubscriptionTier,
    token_data: TokenData = Depends(get_current_user_tier)
):
    """
    Update the subscription tier of the current user
    """
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE users
            SET subscription_tier = %s
            WHERE clerk_id = %s
            RETURNING id, clerk_id, username, email, subscription_tier, created_at
            """,
            (new_tier, token_data.clerk_id)
        )
        updated_user = cursor.fetchone()
        if not updated_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return User(
            id=str(updated_user['id']),
            clerk_id=updated_user['clerk_id'],
            username=updated_user['username'],
            email=updated_user['email'],
            subscription_tier=updated_user['subscription_tier'],
            created_at=str(updated_user['created_at'])
        )

# Event search endpoint
@app.get("/get_events", response_model=List[Event])
def suggest_more(
    topic: Optional[str] = Query(default=None, description="Event Topic"),
    title: Optional[str] = Query(default=None, description="Event Title"),
    yearMin: Optional[int] = Query(default=None, description="Event Min or Near Year"),
    yearMax: Optional[int] = Query(default=None, description="Event Max Year"),
    subjects: Optional[List[str]] = Query(default=None, description="Event Subjects"),
    authorization: Optional[str] = Header(default=None)
):
    # Default values for unauthenticated users
    user_tier = SubscriptionTier.STUDENT
    max_results = 2  # Default for student tier
    use_ai_generation = False  # Default - only available to higher tiers
    
    # Try to get user's tier if authentication is provided
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.split("Bearer ")[1]
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            tier = payload.get("tier")
            if tier:
                user_tier = tier
                
                # Set tier-specific limits
                if user_tier == SubscriptionTier.SCHOLAR:
                    max_results = 5
                    use_ai_generation = True
                elif user_tier == SubscriptionTier.HISTORIAN:
                    max_results = 10
                    use_ai_generation = True
        except:
            # If token validation fails, just use default settings
            pass
    
    # First, try to query the database for matching events
    db_events = []
    
    try:
        query = """
            SELECT id, title, year, lat, lon, subject, info, key_terms
            FROM events
            WHERE 1=1
        """
        params = []
        
        if subjects:
            placeholders = ','.join(['%s'] * len(subjects))
            query += f" AND subject = ANY(ARRAY[{placeholders}])"
            params.extend([s.lower() for s in subjects])
            
        if yearMin is not None:
            query += " AND year >= %s"
            params.append(int(yearMin))
            
        if yearMax is not None:
            query += " AND year <= %s"
            params.append(int(yearMax))
            
        query += " ORDER BY year ASC LIMIT %s"
        params.append(max_results)
        
        with get_db_cursor() as cursor:
            cursor.execute(query, params)
            db_events = cursor.fetchall()
            
        print(f"DB returned {len(db_events)} events")
    except Exception as e:
        print(f"Error querying database: {e}")
    
    # If we have enough DB events based on tier, return them
    if len(db_events) >= max_results:
        return [Event(
            id=ev['id'],
            title=ev['title'],
            year=int(ev['year']),
            lat=ev['lat'],
            lon=ev['lon'],
            subject=ev['subject'],
            info=ev['info'],
            key_terms=ev['key_terms'] if ev['key_terms'] else []
        ) for ev in db_events[:max_results]]
    
    # Otherwise, fall back to Pinecone for advanced tiers
    events = []
    pinecone_results = []
    current_ids = set()
    
    # Add database results to our final list
    for ev in db_events:
        events.append(Event(
            id=ev['id'],
            title=ev['title'],
            year=int(ev['year']),
            lat=ev['lat'],
            lon=ev['lon'],
            subject=ev['subject'],
            info=ev['info'],
            key_terms=ev['key_terms'] if ev['key_terms'] else []
        ))
        current_ids.add(ev['id'])
    
    # If we need more results and we're at Scholar or above, try Pinecone
    if len(events) < max_results and user_tier != SubscriptionTier.STUDENT:
        try:
            if topic:
                normalized_topic = normalize_term(topic)
                embedding_response = client.embeddings.create(input=topic, model="text-embedding-ada-002")
                query_vector = embedding_response.data[0].embedding
                
                # Build filter conditions
                filter_conditions = {normalized_topic: True}
                if yearMin is not None:
                    filter_conditions["year"] = {"$gte": float(yearMin)}
                if yearMax is not None:
                    if "year" in filter_conditions:
                        filter_conditions["year"]["$lte"] = float(yearMax)
                    else:
                        filter_conditions["year"] = {"$lte": float(yearMax)}
                        
                print("Filter conditions (Pinecone):", filter_conditions)
                query_response = index.query(
                    vector=query_vector,
                    top_k=max_results - len(events),
                    include_metadata=True,
                    filter=filter_conditions
                )
                
                print("Pinecone response:", query_response)
                for match in query_response.matches:
                    if match.id not in current_ids:
                        metadata = match.metadata
                        pinecone_results.append(Event(
                            id=match.id,
                            title=metadata.get("title"),
                            year=int(metadata.get("year")),
                            lat=metadata.get("lat"),
                            lon=metadata.get("lon"),
                            subject=metadata.get("subject"),
                            info=metadata.get("info"),
                            key_terms=[]  # We don't store key_terms in metadata
                        ))
                        current_ids.add(match.id)
        except Exception as e:
            print(f"Error retrieving from Pinecone: {e}")
    
    # Add Pinecone results to our final list
    events.extend(pinecone_results)
    
    # If still fewer than max_results and we're allowed to use AI generation, use OpenAI
    if len(events) < max_results and use_ai_generation:
        number_of_events = max_results - len(events)
        print(f"Fetching {number_of_events} more events from OpenAI")
        
        subjects_str = ", ".join([s.capitalize() for s in subjects]) if subjects else "subjects"
        prompt = f"Provide exactly {number_of_events} significant {subjects_str} related to the topic {topic}."
        
        if title:
            prompt += f" Related to the event titled '{title}'."
        if yearMin is not None and yearMax is None:
            prompt += f" That occurred shortly after the year {yearMin}."
        if yearMin is not None and yearMax is not None:
            prompt += f" That occurred between the years {yearMin} and {yearMax}, with a preference for events near the year {yearMin}."
            
        prompt += ("\nPlease respond with a complete and valid JSON array (no extra text), "
                "where each object has the keys: title, year, lat, lon, subject, and info.")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in history and geography, with attention to significant events, technology, science, religion, military efforts, sports, games, and culture, great people and works of art, literature, and music."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )
            
            raw_content = response.choices[0].message.content.strip()
            cleaned_json = re.sub(r"```json|```", "", raw_content).strip()
            suggestions = json.loads(cleaned_json)
            
            for item in suggestions:
                event_id = hashlib.sha256(f"{item['title']}-{item['year']}".encode("utf-8")).hexdigest()
                
                if event_id not in current_ids:
                    key_terms = extract_key_terms(item["title"] + " " + item["info"])
                    event = Event(
                        id=event_id,
                        title=item["title"],
                        year=int(item["year"]),
                        lat=float(item["lat"]),
                        lon=float(item["lon"]),
                        subject=item["subject"].lower(),
                        info=item["info"],
                        key_terms=key_terms
                    )
                    
                    events.append(event)
                    current_ids.add(event_id)
                    
                    # Store the event for future queries
                    store_event(event)
                    
                    if len(events) >= max_results:
                        break
                        
        except Exception as e:
            print(f"Error fetching suggestions: {e}")
    
    print(f"Total events merged for response: {len(events)}")
    return events[:max_results]

# Authentication endpoints
@app.post("/auth/token")
async def create_token(request: Request):
    """
    Create a JWT token for the user with their subscription tier
    Either using query params or request body
    """
    try:
        # Try to get data from request body (JSON)
        request_data = await request.json()
        clerk_id = request_data.get("clerk_id")
        requested_tier = request_data.get("requested_tier")
    except:
        # Fall back to query parameters
        clerk_id = request.query_params.get("clerk_id")
        requested_tier = request.query_params.get("requested_tier")
    
    if not clerk_id:
        raise HTTPException(status_code=400, detail="clerk_id is required")
    
    # Check if user exists
    user = await get_user_from_db(clerk_id)
    
    # If user doesn't exist, create a new one
    if not user:
        try:
            print(f"User with clerk_id {clerk_id} not found, creating new user")
            # Create a basic user with default tier
            new_user = UserCreate(
                clerk_id=clerk_id,
                username=f"user_{clerk_id[-6:]}",  # Use last 6 chars of clerk_id as username
                email=f"{clerk_id}@example.com",   # Placeholder email
                subscription_tier=SubscriptionTier.STUDENT
            )
            
            with get_db_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (clerk_id, username, email, subscription_tier)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, clerk_id, username, email, subscription_tier, created_at
                    """,
                    (new_user.clerk_id, new_user.username, new_user.email, new_user.subscription_tier)
                )
                user_data = cursor.fetchone()
                user = User(
                    id=str(user_data['id']),
                    clerk_id=user_data['clerk_id'],
                    username=user_data['username'],
                    email=user_data['email'],
                    subscription_tier=user_data['subscription_tier'],
                    created_at=str(user_data['created_at'])
                )
                print(f"Created new user: {user}")
        except Exception as e:
            print(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")
    
    # If a specific tier is requested, verify the user has access to it
    if requested_tier and not verify_tier_access(requested_tier, user.subscription_tier):
        raise HTTPException(
            status_code=403, 
            detail=f"User does not have access to the {requested_tier} tier. Current tier: {user.subscription_tier}"
        )
    
    # Create JWT token
    token_data = {
        "clerk_id": user.clerk_id,
        "tier": user.subscription_tier
    }
    
    token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

# Tier-restricted endpoints
@app.get("/scholar/advanced-search")
async def advanced_search(
    query: str,
    token_data: TokenData = Depends(get_current_user_tier)
):
    """
    Advanced search endpoint available to Scholar and Historian tiers
    """
    if not verify_tier_access(SubscriptionTier.SCHOLAR, token_data.tier):
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint requires at least the Scholar tier. Current tier: {token_data.tier}"
        )
    
    # Advanced search implementation would go here
    return {"message": "Advanced search results for query: " + query}

@app.get("/historian/data-analysis")
async def data_analysis(
    dataset: str,
    token_data: TokenData = Depends(get_current_user_tier)
):
    """
    Data analysis endpoint available only to Historian tier
    """
    if not verify_tier_access(SubscriptionTier.HISTORIAN, token_data.tier):
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint requires the Historian tier. Current tier: {token_data.tier}"
        )
    
    # Data analysis implementation would go here
    return {"message": "Data analysis results for dataset: " + dataset}

# Add a health check endpoint
@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok", "version": "1.0.0"}
