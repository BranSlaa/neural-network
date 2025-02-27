from contextlib import contextmanager
from psycopg2.extras import RealDictCursor
import psycopg2
import os

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:brandons-local-server@localhost:5432/history_map_db"

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
            
# Context manager for database cursors
@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
        finally:
            cursor.close()

# Context manager for database connections
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        conn.autocommit = True
        yield conn
    finally:
        conn.close()