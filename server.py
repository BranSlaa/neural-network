import os
import json
import re
import hashlib
from typing import List, Optional, Set
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Initialize OpenAI client.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read configuration from environment.
# For a real embedding model like text-embedding-ada-002, set dimension to 1536.
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "history-map")
pinecone_dim = int(os.getenv("PINECONE_DIMENSION", "1536"))

# Create a Pinecone client instance.
pc = Pinecone(api_key=pinecone_api_key)

# Create FastAPI app.
app = FastAPI()
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Define the event data model.
class Event(BaseModel):
	title: str
	year: int
	lat: float
	lon: float
	subject: str
	info: str

# A helper to compute a cache key (if needed for query caching).
def get_cache_key(topic, title, yearMinOrNear, yearMax, subjects):
	key_data = {
		"topic": topic,
		"title": title,
		"yearMinOrNear": yearMinOrNear,
		"yearMax": yearMax,
		"subjects": subjects,
	}
	return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()

# Global index handle.
index = None

# Initialize a set to track returned event IDs
returned_event_ids: Set[str] = set()

@app.on_event("startup")
async def startup_event():
	global index
	try:
		# Create the index if it doesn't exist.
		if pinecone_index_name not in pc.list_indexes().names():
			pc.create_index(
				name=pinecone_index_name,
				dimension=pinecone_dim,
				metric='cosine',
				spec=ServerlessSpec(cloud='aws', region='us-east-1')
			)
	except Exception as e:
		print(f"Error during index creation: {e}")
	# Retrieve the index handle.
	index = pc.Index(pinecone_index_name)

def get_event_embedding(event: Event) -> List[float]:
	input_text = f"Title: {event.title}. Year: {event.year}. Location: {event.lat}, {event.lon}. Subject: {event.subject}. Info: {event.info}"
	embedding_response = client.embeddings.create(input=input_text, model="text-embedding-ada-002")
	# Access the embedding via dot notation.
	vector = embedding_response.data[0].embedding
	return vector


@app.get("/get_events", response_model=List[Event])
def suggest_more(
	topic: Optional[str] = Query(default=None, description="Event Topic"),
	title: Optional[str] = Query(default=None, description="Event Title"),
	yearMinOrNear: Optional[int] = Query(default=None, description="Event Min or Near Year"),
	yearMax: Optional[int] = Query(default=None, description="Event Max Year"),
	subjects: Optional[List[str]] = Query(default=None, description="Event Subjects"),
):
	"""
	Uses GPT to generate event suggestions.
	First, it attempts to retrieve a cached response from Pinecone.
	If none is found, it calls the API, logs the new response into Pinecone, and returns it.
	"""
	events = []

	# ----- Vectorize the Keywords -----
	try:
		if topic:
			# Create a vector for the topic using OpenAI
			embedding_response = client.embeddings.create(input=topic, model="text-embedding-ada-002")
			query_vector = embedding_response.data[0].embedding

			# ----- Query Pinecone with the Vector -----
			query_response = index.query(
				vector=query_vector,
				top_k=3,  # Limit to 3 results
				include_metadata=True
			)
			results = query_response.matches
			print(f"Found {len(results)} matches for topic '{topic}'.")
			if results:
				for match in results:
					event_id = match.id
					if event_id not in returned_event_ids:
						events.append(Event(**match.metadata))
						returned_event_ids.add(event_id)
						if len(events) == 3:
							break
	except Exception as e:
		print(f"Error retrieving from Pinecone: {e}")

	# If fewer than 3 unique events, fetch more from OpenAI
	if len(events) < 3:
		number_of_events = 3 - len(events)
		print(f"Fetching {number_of_events} more events from OpenAI")
		# ----- Build the Prompt -----
		if subjects:
			subjects_sentence_case = [s.capitalize() for s in subjects]
			if len(subjects_sentence_case) > 1:
				subjects_str = ", ".join(subjects_sentence_case[:-1]) + " or " + subjects_sentence_case[-1]
			else:
				subjects_str = subjects_sentence_case[0]
		else:
			subjects_str = "subjects"

		prompt = f"Provide exactly {number_of_events} significant {subjects_str} related to the topic {topic}."
		if title:
			prompt += f" Related to the event titled '{title}'."
		if yearMinOrNear and not yearMax:
			prompt += f" That occurred shortly after the year {yearMinOrNear}."
		if yearMinOrNear and yearMax:
			prompt += f" That occurred between the years {yearMinOrNear} and {yearMax}, with a preference for events near the year {yearMinOrNear}."

		prompt += """
			Respond in JSON format as a list of objects, each containing:
			- "title" (string) - The title of the event
			- "year" (int) - The year of the event (IF BCE, use negative year)
			- "lat" (float) - The latitude of the event
			- "lon" (float) - The longitude of the event
			- "subject" (string) - The subject of the event
			- "info" (string) - A brief description of the event
		"""

		# ----- Fetch New Data from the GPT API -----
		try:
			response = client.chat.completions.create(
				model="gpt-4o-mini",
				messages=[
					{"role": "system", "content": "You are an expert in history, geography, science, and linguistics."},
					{"role": "user", "content": prompt}
				],
				max_tokens=500,
			)
			raw_content = response.choices[0].message.content.strip()
			cleaned_json = re.sub(r"```json|```", "", raw_content).strip()
			suggestions = json.loads(cleaned_json)

			for item in suggestions:
				event_id = hashlib.sha256(f"{item['title']}-{item['year']}".encode("utf-8")).hexdigest()
				if event_id not in returned_event_ids:
					if isinstance(item, dict) and all(k in item for k in ["title", "year", "lat", "lon", "subject", "info"]):
						events.append(Event(
							title=item["title"],
							year=int(item["year"]),
							lat=float(item["lat"]),
							lon=float(item["lon"]),
							subject=item["subject"],
							info=item["info"]
						))
						returned_event_ids.add(event_id)
						if len(events) == 3:
							break
		except Exception as e:
			print(f"Error fetching suggestions: {e}")

	# Return exactly 3 events
	return events[:3]
