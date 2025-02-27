from fastapi import FastAPI

def setup_routes(app: FastAPI):
    # Add a health check endpoint
    @app.get("/health")
    async def health_check():
        """
        Simple health check endpoint
        """
        return {"status": "ok", "version": "1.0.0"}

    @app.get("/")
    async def root():
        """
        Root endpoint
        """
        return {"message": "Welcome to the History Map API!"}
