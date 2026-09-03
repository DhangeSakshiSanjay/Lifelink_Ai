from fastapi import FastAPI

app = FastAPI(title="Lifelink-AI Backend")


@app.get("/")
def home():
    return {
        "message": "Lifelink-AI Backend is running"
    }