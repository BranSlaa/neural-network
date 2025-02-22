from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import json
import re


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Enable CORS
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Allow all origins (change in production)
	allow_credentials=True,
	allow_methods=["*"],  # Allow all HTTP methods
	allow_headers=["*"],  # Allow all headers
)

class Event(BaseModel):
	title: str
	year: int
	lat: float
	lon: float
	subject: str
	info: str

@app.get("/get_events", response_model=List[Event])
def suggest_more(
    topic: Optional[str] = Query(default=None, description="Event Topic"),
    title: Optional[str] = Query(default=None, description="Event Title"),
    yearMinOrNear: Optional[int] = Query(default=None, description="Event Min or Near Year"),
    yearMax: Optional[int] = Query(default=None, description="Event Max Year"),
    subjects: Optional[List[str]] = Query(default=None, description="Event Subjects"),
):
    """
    Uses OpenAI GPT model to suggest related subjects based on optional filters.
    """
    
    if subjects:
        subjects_sentence_case = [subject.capitalize() for subject in subjects]
        if len(subjects_sentence_case) > 1:
            subjects_str = ", ".join(subjects_sentence_case[:-1]) + " or " + subjects_sentence_case[-1]
        else:
            subjects_str = subjects_sentence_case[0]
    else:
        subjects_str = "subjects"

    prompt = f"Provide exactly three significant {subjects_str} related to the topic {topic}."

    if title:
        prompt += f" Related to the event titled '{title}'."
    if yearMinOrNear and not yearMax:
        prompt += f" That occurred shortly after the year {yearMinOrNear}."
    if yearMinOrNear and yearMax:
        prompt += f" That occurred between the years {yearMinOrNear} and {yearMax}, with a preference for events that occurred near the year {yearMinOrNear}."

    prompt += """
    Respond in JSON format as a list of objects, each containing:
    - "title" (string)
    - "year" (int)
    - "lat" (float)
    - "lon" (float)
    - "subject" (string)
    - "info" (string)
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in history, geography, science, and linguistics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
        )

        raw_content = response.choices[0].message.content.strip()
        
        cleaned_json = re.sub(r"```json|```", "", raw_content).strip()
        suggestions = json.loads(cleaned_json)

        locations = []
        for item in suggestions:
            if isinstance(item, dict) and all(k in item for k in ["title", "year", "lat", "lon", "subject", "info"]):
                locations.append(Event(
                    title=item["title"],
                    year=int(item["year"]),
                    lat=float(item["lat"]),
                    lon=float(item["lon"]),
                    subject=item["subject"],
                    info=item["info"]
                ))

        return locations[:3]

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return []
    except Exception as e:
        print(f"Error fetching suggestions: {e}")
        return []