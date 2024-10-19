from datetime import datetime
from bson import ObjectId

class User:
    def __init__(self, name, email, password, country, liked_posts=None, bookmarked_posts=None):
        self.name = name
        self.email = email
        self.password = password
        self.country = country
        self.liked_posts = liked_posts or []  # Array of post IDs
        self.bookmarked_posts = bookmarked_posts or []  # Array of post IDs
        self.created_at = datetime.utcnow()

    def to_dict(self):
        return {
            "name": self.name,
            "email": self.email,
            "password": self.password,
            "country": self.country,
            "liked_posts": self.liked_posts,
            "bookmarked_posts": self.bookmarked_posts,
            "created_at": self.created_at
        }

    @staticmethod
    def from_dict(data):
        return User(
            name=data.get('name'),
            email=data.get('email'),
            password=data.get('password'),
            country=data.get('country'),
            liked_posts=data.get('liked_posts', []),
            bookmarked_posts=data.get('bookmarked_posts', [])
        )