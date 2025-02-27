from functions.database import get_db_cursor

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