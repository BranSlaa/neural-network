from fastapi import HTTPException, Depends, FastAPI
from classes.classes import User, UserCreate, TokenData, SubscriptionTier
from functions.functions import get_current_user_tier, get_user_from_db
from functions.database import get_db_cursor

def setup_routes(app: FastAPI):
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