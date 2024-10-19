# articles.py

from flask import Blueprint, jsonify, request
from bson.json_util import dumps
from config.config import Config
from models.article_model import save_article_to_db

articles_bp = Blueprint('articles', __name__)

@articles_bp.route('/api/articles', methods=['GET'])
def get_articles():
    try:
        articles = request.mongo.db.articles.find()  # Sort by latest
        articles_list = list(articles)  # Convert cursor to list
        return dumps(articles_list), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
