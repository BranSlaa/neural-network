from fastapi import HTTPException, Depends, FastAPI
from typing import List, Dict, Optional
from classes.classes import User, TokenData, SubscriptionTier
from functions.functions import get_current_user_tier, get_user_from_db
from functions.database import get_db_cursor

# Basic quiz model (add to classes.py later)
class Quiz:
    def __init__(self, id: str, title: str, description: str, questions: List[Dict] = None, 
                difficulty: str = "beginner", subject: Optional[str] = None):
        self.id = id
        self.title = title
        self.description = description
        self.questions = questions or []
        self.difficulty = difficulty
        self.subject = subject
        
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "questions": self.questions,
            "difficulty": self.difficulty,
            "subject": self.subject
        }

def setup_routes(app: FastAPI):
    @app.get("/quizzes")
    async def get_quizzes(token_data: TokenData = Depends(get_current_user_tier)):
        """
        Get a list of available quizzes based on user's subscription tier
        """
        # Sample quizzes - in production, these would come from a database
        quizzes = [
            Quiz(
                id="quiz1",
                title="Introduction to Ancient Greece",
                description="Test your knowledge about Ancient Greek civilization",
                difficulty="beginner",
                subject="Ancient Greece"
            ),
            Quiz(
                id="quiz2",
                title="Medieval Europe Challenge",
                description="Advanced questions about Medieval European history",
                difficulty="intermediate",
                subject="Medieval History"
            ),
            Quiz(
                id="quiz3",
                title="World War II: Major Events",
                description="Test your knowledge of key WWII events",
                difficulty="advanced",
                subject="Modern History"
            )
        ]
        
        # Return all quizzes for now - in a real implementation, filter based on user tier
        return [quiz.to_dict() for quiz in quizzes]
        
    @app.get("/quizzes/{quiz_id}")
    async def get_quiz_by_id(quiz_id: str, token_data: TokenData = Depends(get_current_user_tier)):
        """
        Get a specific quiz by its ID
        """
        # In a real implementation, fetch from database
        # For now, return a dummy quiz
        if quiz_id == "quiz1":
            quiz = Quiz(
                id="quiz1",
                title="Introduction to Ancient Greece",
                description="Test your knowledge about Ancient Greek civilization",
                difficulty="beginner",
                subject="Ancient Greece",
                questions=[
                    {
                        "id": 1,
                        "question": "Which city-state is known for its military prowess and strict social system?",
                        "options": ["Athens", "Sparta", "Corinth", "Thebes"],
                        "answer": 1  # Index of the correct answer (Sparta)
                    },
                    {
                        "id": 2,
                        "question": "Who was the legendary king of Ithaca who fought in the Trojan War?",
                        "options": ["Achilles", "Hector", "Odysseus", "Agamemnon"],
                        "answer": 2  # Odysseus
                    }
                ]
            )
            return quiz.to_dict()
        else:
            raise HTTPException(status_code=404, detail="Quiz not found") 