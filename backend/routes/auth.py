from flask import Blueprint, request, jsonify
import jwt
from datetime import datetime, timedelta
from bson import ObjectId
from models.user import User
from utils.password_utils import hash_password, check_password
from config.config import Config

auth = Blueprint('auth', __name__)

@auth.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'email', 'password', 'country']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    "success": False, 
                    "message": f"{field} is required"
                }), 400

        # Check if user already exists
        if request.mongo.db.users.find_one({"email": data['email']}):
            return jsonify({
                "success": False, 
                "message": "Email already registered"
            }), 400

        # Create new user
        hashed_password = hash_password(data['password'])
        user = User(
            name=data['name'],
            email=data['email'],
            password=hashed_password,
            country=data['country'],
            liked_posts=[],
            bookmarked_posts=[]
        )
        
        request.mongo.db.users.insert_one(user.to_dict())
        
        return jsonify({
            "success": True, 
            "message": "User registered successfully"
        }), 201
    
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500

@auth.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        user = request.mongo.db.users.find_one({"email": data['email']})
        
        if not user or not check_password(data['password'], user['password']):
            return jsonify({
                "success": False, 
                "message": "Invalid email or password"
            }), 401

        # Generate JWT token
        token = jwt.encode({
            'user_id': str(user['_id']),
            'exp': datetime.utcnow() + timedelta(days=1)
        }, Config.JWT_SECRET_KEY)

        return jsonify({
            "success": True,
            "token": token,
            "user": {
                "id": str(user['_id']),
                "name": user['name'],
                "email": user['email'],
                "country": user['country'],
                "liked_posts": user.get('liked_posts', []),
                "bookmarked_posts": user.get('bookmarked_posts', [])
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500

@auth.route('/user/likes', methods=['POST'])
def toggle_like():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                "success": False, 
                "message": "No token provided"
            }), 401

        # Decode token
        payload = jwt.decode(
            token.split(" ")[1], 
            Config.JWT_SECRET_KEY, 
            algorithms=["HS256"]
        )
        user_id = payload['user_id']
        
        data = request.get_json()
        post_id = data.get('post_id')
        
        if not post_id:
            return jsonify({
                "success": False, 
                "message": "Post ID is required"
            }), 400

        user = request.mongo.db.users.find_one({"_id": ObjectId(user_id)})
        liked_posts = user.get('liked_posts', [])

        # Toggle like
        if post_id in liked_posts:
            liked_posts.remove(post_id)
        else:
            liked_posts.append(post_id)

        # Update user
        request.mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"liked_posts": liked_posts}}
        )

        return jsonify({
            "success": True,
            "liked_posts": liked_posts
        }), 200

    except jwt.ExpiredSignatureError:
        return jsonify({
            "success": False, 
            "message": "Token has expired"
        }), 401
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500

@auth.route('/user/bookmarks', methods=['POST'])
def toggle_bookmark():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                "success": False, 
                "message": "No token provided"
            }), 401

        # Decode token
        payload = jwt.decode(
            token.split(" ")[1], 
            Config.JWT_SECRET_KEY, 
            algorithms=["HS256"]
        )
        user_id = payload['user_id']
        
        data = request.get_json()
        post_id = data.get('post_id')
        
        if not post_id:
            return jsonify({
                "success": False, 
                "message": "Post ID is required"
            }), 400

        user = request.mongo.db.users.find_one({"_id": ObjectId(user_id)})
        bookmarked_posts = user.get('bookmarked_posts', [])

        # Toggle bookmark
        if post_id in bookmarked_posts:
            bookmarked_posts.remove(post_id)
        else:
            bookmarked_posts.append(post_id)

        # Update user
        request.mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"bookmarked_posts": bookmarked_posts}}
        )

        return jsonify({
            "success": True,
            "bookmarked_posts": bookmarked_posts
        }), 200

    except jwt.ExpiredSignatureError:
        return jsonify({
            "success": False, 
            "message": "Token has expired"
        }), 401
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500

@auth.route('/user/profile', methods=['GET'])
def get_profile():
    try:
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({
                "success": False, 
                "message": "No token provided"
            }), 401

        # Decode token
        payload = jwt.decode(
            token.split(" ")[1], 
            Config.JWT_SECRET_KEY, 
            algorithms=["HS256"]
        )
        user_id = payload['user_id']

        user = request.mongo.db.users.find_one({"_id": ObjectId(user_id)})
        if not user:
            return jsonify({
                "success": False, 
                "message": "User not found"
            }), 404

        return jsonify({
            "success": True,
            "user": {
                "id": str(user['_id']),
                "name": user['name'],
                "email": user['email'],
                "country": user['country'],
                "liked_posts": user.get('liked_posts', []),
                "bookmarked_posts": user.get('bookmarked_posts', [])
            }
        }), 200

    except jwt.ExpiredSignatureError:
        return jsonify({
            "success": False, 
            "message": "Token has expired"
        }), 401
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": str(e)
        }), 500
    