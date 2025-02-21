from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from openai import OpenAI
import os
import json
import re
from functools import lru_cache


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

# Set up OpenAI API key (ensure you set this in your environment)

class Location(BaseModel):
	name: str
	lat: float
	lon: float
	info: str

# Dummy database of locations with topics
topic_locations = {
	"history": [
		{"name": "Colosseum", "lat": 41.8902, "lon": 12.4922, "info": "Ancient Roman amphitheater."},
		{"name": "Great Wall of China", "lat": 40.4319, "lon": 116.5704, "info": "Historic fortifications in China."},
		{"name": "Machu Picchu", "lat": -13.1631, "lon": -72.5450, "info": "Incan citadel in Peru."}
	],
	"art": [
		{"name": "Louvre Museum", "lat": 48.8606, "lon": 2.3376, "info": "Home of the Mona Lisa."},
		{"name": "MoMA", "lat": 40.7614, "lon": -73.9776, "info": "Museum of Modern Art in NYC."},
		{"name": "Uffizi Gallery", "lat": 43.7687, "lon": 11.2560, "info": "Renaissance masterpieces in Florence."}
	]
}

@app.get("/get_locations", response_model=List[Location])
def get_locations(topic: str = Query(..., description="Topic of interest")):
	if topic.lower() in topic_locations:
		return topic_locations[topic.lower()]
	return []

@app.get("/suggest_more", response_model=List[Location])
def suggest_more(name: str = Query(..., description="Selected location name")):
	"""
	Uses OpenAI GPT model to suggest locations related to the selected place with historical, geographical,
	scientific, and linguistic relevance.
	"""
	prompt = f"""
	Provide exactly three historically, geographically, scientifically, or linguistically significant locations related to {name}.
	Respond in JSON format as a list of objects, each containing:
	- "name" (string)
	- "lat" (float)
	- "lon" (float)
	- "info" (string)
	"""

	try:
		response = client.chat.completions.create(
			model="gpt-4o",  # Updated to use GPT-4o
			messages=[
				{"role": "system", "content": "You are an expert in history, geography, science, and linguistics."},
				{"role": "user", "content": prompt}
			],
			max_tokens=500,
		)
		
		# Extract response content
		raw_content = response.choices[0].message.content
		print("Raw OpenAI Response:", raw_content)  # Debugging

		# Strip markdown formatting if present
		cleaned_json = re.sub(r"```json|```", "", raw_content).strip()

		# Parse JSON safely
		suggestions = json.loads(cleaned_json)

		# Validate response format
		locations = []
		for item in suggestions:
			if isinstance(item, dict) and "name" in item and "lat" in item and "lon" in item and "info" in item:
				locations.append(Location(
					name=item["name"],
					lat=float(item["lat"]),
					lon=float(item["lon"]),
					info=item["info"]
				))

		return locations[:3]  # Ensure only three results are returned
	
	except json.JSONDecodeError as e:
		print(f"JSON parsing error: {e}")
		return []
	except Exception as e:
		print(f"Error fetching suggestions: {e}")
		return []
	
@lru_cache(maxsize=50)
def get_locations_cached(topic: str):
    return get_locations(topic)