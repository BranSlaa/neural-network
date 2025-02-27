from typing import List, Optional
from fastapi import Query, Header, FastAPI
from classes.classes import Event, SubscriptionTier
import jwt
import hashlib
import re
import json
import os
from functions.database import get_db_cursor
from functions.functions import normalize_term, extract_key_terms, store_event
from openai import OpenAI

# Global variables to be set from server.py
index = None

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def setup_routes(app: FastAPI):
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
