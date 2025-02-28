from classes.classes import SubscriptionTier
import jwt
from functions.database import get_db_cursor
from functions.functions import get_user_from_db, verify_tier_access
from fastapi import FastAPI, HTTPException, Request
from classes.classes import User, UserCreate, SubscriptionTier
import os

JWT_SECRET = os.getenv("JWT_SECRET") or "9357048b73a03490c56e4d830d6fb60cf9d9352c9e6b34aa647ea80aada247d8"
JWT_ALGORITHM = "HS256"

# Create a router
from fastapi import APIRouter
router = APIRouter()

@router.post("/auth/token")
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
                subscription_tier=SubscriptionTier.STUDENT  # Value is 1
            )
            
            with get_db_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (clerk_id, username, email, subscription_tier)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, clerk_id, username, email, subscription_tier, created_at
                    """,
                    (new_user.clerk_id, new_user.username, new_user.email, new_user.subscription_tier.value)  # Use numeric value
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
        "tier": user.subscription_tier.value  # Store numeric value (1, 2, or 3)
    }
    
    # Print token data for debugging
    print(f"Creating token with data: {token_data}")
    
    token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

def setup_routes(app: FastAPI):
    # Include the router
    app.include_router(router)