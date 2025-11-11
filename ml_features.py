from huggingface_hub import InferenceClient
import os
from models import MLFeature, Anime, Review
from database import get_session
import json

def get_hf_client():
    token = os.environ.get('HUGGINGFACE_TOKEN') or os.environ.get('HF_TOKEN')
    if token:
        return InferenceClient(token=token)
    return InferenceClient()

def classify_synopsis(synopsis):
    if not synopsis or len(synopsis.strip()) < 10:
        return "Unknown"
    
    try:
        client = get_hf_client()
        
        result = client.text_classification(
            synopsis[:512],
            model="facebook/bart-large-mnli"
        )
        
        categories = ["Action", "Comedy", "Drama", "Romance", "Mystery", "Fantasy", "Sci-Fi"]
        
        predictions = []
        for category in categories:
            hypothesis = f"This is a {category} anime."
            score_result = client.text_classification(
                f"{synopsis[:200]}. {hypothesis}",
                model="facebook/bart-large-mnli"
            )
            if score_result and len(score_result) > 0:
                predictions.append((category, score_result[0].get('score', 0)))
        
        if predictions:
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[0][0]
        
        return "General"
        
    except Exception as e:
        print(f"Error classifying synopsis: {e}")
        return "Unknown"

def analyze_sentiment(text):
    if not text or len(text.strip()) < 10:
        return 0.5
    
    try:
        client = get_hf_client()
        
        result = client.text_classification(
            text[:512],
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        if result and len(result) > 0:
            label = result[0].get('label', 'NEUTRAL').upper()
            score = result[0].get('score', 0.5)
            
            if label == 'POSITIVE':
                return score
            elif label == 'NEGATIVE':
                return 1.0 - score
            else:
                return 0.5
        
        return 0.5
        
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.5

def generate_synopsis_embedding(synopsis):
    if not synopsis or len(synopsis.strip()) < 10:
        return None
    
    try:
        client = get_hf_client()
        
        result = client.feature_extraction(
            synopsis[:512],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        if result and len(result) > 0:
            if isinstance(result[0], list):
                embedding = result[0][:128]
            else:
                embedding = result[:128]
            
            return json.dumps(embedding)
        
        return None
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_anime_ml_features(anime_id):
    session = get_session()
    try:
        anime = session.query(Anime).filter_by(id=anime_id).first()
        if not anime:
            return False
        
        existing_ml = session.query(MLFeature).filter_by(anime_id=anime_id).first()
        
        synopsis_category = None
        synopsis_embedding = None
        
        if anime.synopsis:
            print(f"Classifying synopsis for: {anime.title}")
            synopsis_category = classify_synopsis(anime.synopsis)
            synopsis_embedding = generate_synopsis_embedding(anime.synopsis)
        
        if existing_ml:
            existing_ml.synopsis_category = synopsis_category
            existing_ml.synopsis_embedding = synopsis_embedding
            existing_ml.predicted_rating = anime.score
        else:
            ml_feature = MLFeature(
                anime_id=anime_id,
                synopsis_category=synopsis_category,
                synopsis_embedding=synopsis_embedding,
                predicted_rating=anime.score
            )
            session.add(ml_feature)
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        print(f"Error processing ML features: {e}")
        return False
    finally:
        session.close()

def process_review_sentiment(review_id):
    session = get_session()
    try:
        review = session.query(Review).filter_by(id=review_id).first()
        if not review or not review.review_text:
            return False
        
        print(f"Analyzing sentiment for review {review_id}")
        sentiment_score = analyze_sentiment(review.review_text)
        review.sentiment_score = sentiment_score
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        print(f"Error processing review sentiment: {e}")
        return False
    finally:
        session.close()

def batch_process_ml_features(limit=10):
    session = get_session()
    try:
        anime_without_ml = session.query(Anime).outerjoin(MLFeature).filter(
            MLFeature.id == None,
            Anime.synopsis != None
        ).limit(limit).all()
        
        results = {
            'processed': 0,
            'failed': 0
        }
        
        for anime in anime_without_ml:
            success = process_anime_ml_features(anime.id)
            if success:
                results['processed'] += 1
            else:
                results['failed'] += 1
        
        return results
        
    finally:
        session.close()
