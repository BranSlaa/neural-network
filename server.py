from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
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

class Event(BaseModel):
	title: str
	year: int
	lat: float
	lon: float
	subject: str
	info: str

events = [
	{
		"title": "Colosseum",
		"year": 80,
		"lat": 41.8902,
		"lon": 12.4922,
		"subject": "architecture",
		"info": "The Colosseum, also known as the Flavian Amphitheater, was inaugurated by Emperor Titus in 80 AD with 100 days of games, including gladiatorial contests and wild animal fights."
	},
	{
		"title": "Colosseum",
		"year": 217,
		"lat": 41.8902,
		"lon": 12.4922,
		"subject": "architecture",
		"info": "A major fire in 217 AD severely damaged the upper levels of the Colosseum, causing structural weaknesses that were later repaired under Emperor Alexander Severus."
	},
	{
		"title": "Colosseum",
		"year": 1349,
		"lat": 41.8902,
		"lon": 12.4922,
		"subject": "architecture",
		"info": "A powerful earthquake in 1349 caused the southern outer wall of the Colosseum to collapse, leading to its materials being repurposed for building projects throughout Rome."
	},
	{
		"title": "Great Wall of China",
		"year": -221,
		"lat": 40.4319,
		"lon": 116.5704,
		"subject": "architecture",
		"info": "The Great Wall of China was initiated under Emperor Qin Shi Huang in 221 BCE to protect against northern invaders, using forced labor and connecting pre-existing fortifications."
	},
	{
		"title": "Great Wall of China",
		"year": 1368,
		"lat": 40.4319,
		"lon": 116.5704,
		"subject": "architecture",
		"info": "During the Ming Dynasty (1368), the Great Wall was extensively rebuilt and reinforced with bricks and stone, making it the longest and most durable version of the structure."
	},
	{
		"title": "Great Wall of China",
		"year": 1644,
		"lat": 40.4319,
		"lon": 116.5704,
		"subject": "military",
		"info": "By 1644, after the fall of the Ming Dynasty, the Great Wall lost its military significance as the Manchus successfully invaded China, leading to the establishment of the Qing Dynasty."
	},
	{
		"title": "Machu Picchu",
		"year": 1450,
		"lat": -13.1631,
		"lon": -72.5450,
		"subject": "architecture",
		"info": "Machu Picchu was constructed around 1450 as an estate for the Inca Emperor Pachacuti, featuring sophisticated agricultural terraces, temples, and residences."
	},
	{
		"title": "Machu Picchu",
		"year": 1572,
		"lat": -13.1631,
		"lon": -72.5450,
		"subject": "colonization",
		"info": "Following the Spanish conquest in 1572, Machu Picchu was abandoned and remained largely unknown to outsiders for centuries, avoiding Spanish destruction."
	},
	{
		"title": "Machu Picchu",
		"year": 1911,
		"lat": -13.1631,
		"lon": -72.5450,
		"subject": "archaeology",
		"info": "American historian Hiram Bingham rediscovered Machu Picchu in 1911, bringing international attention to the site and leading to ongoing archaeological studies."
	},
	{
		"title": "Stonehenge",
		"year": -3000,
		"lat": 51.1789,
		"lon": -1.8262,
		"subject": "religion",
		"info": "Stonehenge began construction around 3000 BCE as a circular earthwork enclosure, possibly serving as a ceremonial or astronomical site."
	},
	{
		"title": "Stonehenge",
		"year": -2500,
		"lat": 51.1789,
		"lon": -1.8262,
		"subject": "religion",
		"info": "By 2500 BCE, the final arrangement of massive sarsen stones was erected, aligned with the solstices, indicating advanced knowledge of celestial cycles."
	},
	{
		"title": "Stonehenge",
		"year": 1986,
		"lat": 51.1789,
		"lon": -1.8262,
		"subject": "UNESCO heritage",
		"info": "In 1986, Stonehenge was declared a UNESCO World Heritage Site, ensuring its protection and ongoing research into its origins and cultural significance."
	},
	{
		"title": "Acropolis of Athens",
		"year": -447,
		"lat": 37.9715,
		"lon": 23.7264,
		"subject": "architecture",
		"info": "The Acropolis of Athens saw the beginning of the Parthenon's construction in 447 BCE, dedicated to the goddess Athena and showcasing Doric architectural brilliance."
	},
	{
		"title": "Acropolis of Athens",
		"year": 1687,
		"lat": 37.9715,
		"lon": 23.7264,
		"subject": "military",
		"info": "In 1687, the Parthenon suffered extensive damage when a Venetian bombardment ignited Ottoman gunpowder stored inside, causing part of its structure to collapse."
	},
	{
		"title": "Acropolis of Athens",
		"year": 1975,
		"lat": 37.9715,
		"lon": 23.7264,
		"subject": "restoration",
		"info": "Major restoration efforts for the Acropolis began in 1975 to preserve the ancient ruins, utilizing modern technology to carefully reconstruct damaged sections."
	},
	{
		"title": "Eiffel Tower",
		"year": 1889,
		"lat": 48.8584,
		"lon": 2.2945,
		"subject": "architecture",
		"info": "The Eiffel Tower was inaugurated in 1889 as the centerpiece of the Exposition Universelle (World's Fair), showcasing France's engineering prowess."
	},
	{
		"title": "Eiffel Tower",
		"year": 1940,
		"lat": 48.8584,
		"lon": 2.2945,
		"subject": "military",
		"info": "During World War II, Nazi forces occupied Paris and raised the swastika over the Eiffel Tower, though the French resistance cut its lift cables to prevent enemy use."
	},
	{
		"title": "Eiffel Tower",
		"year": 1985,
		"lat": 48.8584,
		"lon": 2.2945,
		"subject": "architecture",
		"info": "A major renovation of the Eiffel Tower took place in 1985, including structural reinforcement and a complete repainting to maintain its iconic appearance."
	},
	{
		"title": "Hiroshima",
		"year": 1945,
		"lat": 34.3853,
		"lon": 132.4553,
		"subject": "military",
		"info": "On August 6, 1945, the United States dropped the first atomic bomb on Hiroshima, instantly killing tens of thousands and leading to the end of World War II."
	},
	{
		"title": "Hiroshima",
		"year": 1949,
		"lat": 34.3853,
		"lon": 132.4553,
		"subject": "peace movement",
		"info": "In 1949, Hiroshima was declared a City of Peace, committed to nuclear disarmament and serving as a global symbol of peace and reconciliation."
	},
	{
		"title": "Hiroshima",
		"year": 1954,
		"lat": 34.3853,
		"lon": 132.4553,
		"subject": "memorial",
		"info": "Hiroshima Peace Memorial Park was established in 1954, featuring the Atomic Bomb Dome and numerous monuments dedicated to the victims and survivors."
	},
	{
		"title": "Petra",
		"year": -312,
		"lat": 30.3285,
		"lon": 35.4444,
		"subject": "architecture",
		"info": "Petra was founded in 312 BCE by the Nabataeans as a crucial trade hub connecting Arabia, Egypt, and the Mediterranean, flourishing due to its advanced water management system."
	},
	{
		"title": "Petra",
		"year": 363,
		"lat": 30.3285,
		"lon": 35.4444,
		"subject": "natural disasters",
		"info": "In 363 CE, a powerful earthquake destroyed much of Petra's infrastructure, contributing to its decline as trade routes shifted."
	},
	{
		"title": "Petra",
		"year": 1812,
		"lat": 30.3285,
		"lon": 35.4444,
		"subject": "archaeology",
		"info": "Swiss explorer Johann Burckhardt rediscovered Petra in 1812, bringing the lost city to Western attention and sparking archaeological interest."
	},
	{
		"title": "Angkor Wat",
		"year": 1122,
		"lat": 13.4125,
		"lon": 103.8667,
		"subject": "religion",
		"info": "Angkor Wat was built in 1122 under King Suryavarman II as a Hindu temple dedicated to Vishnu, later transforming into a Buddhist site."
	},
	{
		"title": "Angkor Wat",
		"year": 1431,
		"lat": 13.4125,
		"lon": 103.8667,
		"subject": "abandonment",
		"info": "The Khmer Empire fell in 1431, leading to the abandonment of Angkor Wat, with the jungle reclaiming much of the once-great temple complex."
	},
	{
		"title": "Angkor Wat",
		"year": 1860,
		"lat": 13.4125,
		"lon": 103.8667,
		"subject": "archaeology",
		"info": "French explorer Henri Mouhot rediscovered Angkor Wat in 1860, calling it 'grander than anything left to us by Greece or Rome' and reviving interest in its preservation."
	},
	{
		"title": "Alhambra",
		"year": 889,
		"lat": 37.1761,
		"lon": -3.5881,
		"subject": "architecture",
		"info": "The Alhambra was initially constructed as a fortress in 889 AD and later expanded into a grand Islamic palace by the Nasrid Dynasty in the 13th century."
	},
	{
		"title": "Alhambra",
		"year": 1333,
		"lat": 37.1761,
		"lon": -3.5881,
		"subject": "architecture",
		"info": "In 1333, Sultan Yusuf I transformed the Alhambra into a royal palace, featuring intricate Moorish architecture and elaborate gardens."
	},
	{
		"title": "Alhambra",
		"year": 1492,
		"lat": 37.1761,
		"lon": -3.5881,
		"subject": "military",
		"info": "After the Christian Reconquista, the Catholic Monarchs took control of Granada in 1492, converting parts of the Alhambra into a Spanish royal palace."
	}
]

# # Cached event database
# @lru_cache(maxsize=50)
# def get_events_cached(topic: Optional[str] = None):
#     if not topic:
#         return events  # Return all events if no topic is given
#     return [
#         event for event in events
#         if topic.lower() in event["info"].lower() or topic.lower() in event["title"].lower()
#     ]

# @app.get("/get_events", response_model=List[Event])
# async def get_events(topic: Optional[str] = Query(None, description="Event Topic")):
#     print(f"Fetching events for topic: {topic}")  # Debugging log
#     return get_events_cached(topic)

@app.get("/get_events", response_model=List[Event])
def suggest_more(
    title: Optional[str] = Query(default=None, description="Event Title"),
    year: Optional[int] = Query(default=None, description="Event Year"),
    topic: Optional[str] = Query(default=None, description="Event Topic")
):
    """
    Uses OpenAI GPT model to suggest related locations or events based on optional filters.
    """
    prompt = "Provide exactly three historically, geographically, scientifically, or linguistically significant locations."

    if title:
        prompt += f" Related to the event titled '{title}'."
    if year:
        prompt += f" That occurred shortly after the year {year}."
    if topic:
        prompt += f" Consider the topic '{topic}' for relevance."

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