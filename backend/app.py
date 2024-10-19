from flask import Flask, request
from flask_cors import CORS
from pymongo import MongoClient
from routes.auth import auth
from routes.articles import articles_bp  # Import the articles blueprint
from config.config import Config
from scraper.aljazeera_scraper import run_scraper
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# MongoDB connection
client = MongoClient(Config.MONGO_URI)
db = client.get_database('news_aggregator')

@app.before_request
def before_request():
    request.mongo = client
    request.mongo.db = db

app.register_blueprint(auth, url_prefix='/api/auth')
app.register_blueprint(articles_bp)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=lambda: run_scraper(db), trigger="interval", hours=1)
    scheduler.start()

if __name__ == '__main__':
    run_scraper(db)  # Trigger scraper on startup and pass `db`
    start_scheduler()  # Start background scheduler
    app.run(debug=True)
