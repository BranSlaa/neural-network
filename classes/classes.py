from typing import List, Optional
from enum import Enum
from pydantic import BaseModel

class SubscriptionTier(int, Enum):
    STUDENT = 1
    SCHOLAR = 2
    HISTORIAN = 3

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