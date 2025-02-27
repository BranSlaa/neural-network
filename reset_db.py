import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://postgres:brandons-local-server@localhost:5432/history_map_db"

# Context manager for database connections
@contextmanager
def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    try:
        conn.autocommit = True
        yield conn
    finally:
        conn.close()

# Context manager for database cursors
@contextmanager
def get_db_cursor():
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            yield cursor
        finally:
            cursor.close()

def drop_all_tables():
    """Drop all tables in the database"""
    with get_db_cursor() as cursor:
        cursor.execute("""
            DO $$ DECLARE
                r RECORD;
            BEGIN
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = current_schema()) LOOP
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                END LOOP;
            END $$;
        """)
        print("All tables dropped!")

if __name__ == "__main__":
    drop_all_tables()
    print("Database reset completed. Please restart your server to recreate tables.") 