from app.database.connection import engine
from app.models.user import User
from sqlalchemy import select


with engine.connect() as connection:
    result = connection.execute(select(User))
    users = result.fetchall()

    print("Users table connected successfully!")
    print("Number of users:", len(users))