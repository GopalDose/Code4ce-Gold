def save_article_to_db(article, db):
    """Save a single article to the MongoDB database."""
    # Check if the article already exists based on the link
    existing_article = db.articles.find_one({'link': article['link']})
    
    if not existing_article:
        db.articles.insert_one(article)
        print(f"Article saved: {article['title']}")
    else:
        print(f"Article already exists: {article['title']}")
