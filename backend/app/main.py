from sqlalchemy import text
from app.database.connection import engine
from fastapi import FastAPI

app = FastAPI(
    title="Lifelink-AI",
    description="AI-Driven Smart Organ Matching and Transportation Decision Support System",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Lifelink-AI Backend is running successfully"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.get("/db-test")
def database_test():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar()

        return {
            "status": "success",
            "database": "connected",
            "test_value": value
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }