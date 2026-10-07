import os
BASE_DIR=os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
class Config:
    SECRET_KEY=os.getenv('SECRET_KEY','lifelink-demo-secret-change-me')
    SQLALCHEMY_DATABASE_URI='sqlite:///'+os.path.join(BASE_DIR,'instance','lifelink.db')
    SQLALCHEMY_TRACK_MODIFICATIONS=False
    ROUTE_MODE=os.getenv('ROUTE_MODE','static')
    ROUTING_API_KEY=os.getenv('ROUTING_API_KEY','')
