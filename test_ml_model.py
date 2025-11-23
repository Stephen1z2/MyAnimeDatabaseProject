#!/usr/bin/env python3
"""
Use the trained ML model to find hidden gems - underrated anime that should be highly rated
"""

from anime_ml_advanced import AnimeMLAnalyzer
from database import get_session
from models import Anime
import pandas as pd
import numpy as np

def find_hidden_gems():
    """Use ML model to find potentially underrated anime (hidden gems)"""
    
    print("💎 FINDING HIDDEN GEMS WITH MACHINE LEARNING")
    print("=" * 60)
    print("Using your trained ML model to find underrated anime...")
    
    # Initialize and train ML module
    ml = AnimeMLAnalyzer()
    print("📊 Loading data and training score prediction model...")
    ml.load_and_prepare_data()
    
    # Train the score predictor
    results = ml.train_score_predictor(test_size=0.2)
    
    if not results:
        print("❌ Could not train model")
        return
    
    print(f"✅ Model trained! R² Score: {results.get('r2_score', 'Unknown'):.3f}")
    
    # Get the trained model
    if 'score_predictor' not in ml.models:
        print("❌ Score predictor not found in trained models")
        return
    
    model, scaler = ml.models['score_predictor']
    session = get_session()
    
    # Get anime with scores for comparison
    all_anime = session.query(Anime).filter(
        Anime.score.isnot(None),
        Anime.episodes.isnot(None)
    ).all()
    
    print(f"🔍 Analyzing {len(all_anime)} anime for hidden gems...")
    
    # Prepare features and make predictions
    hidden_gems = []
    
    for anime in all_anime:
        # Extract features for this anime
        features = ml._extract_anime_features(anime)
        
        # Prepare features for prediction (same as training)
        feature_vector = np.array([
            features['episodes'],
            features['score_members'] if 'score_members' in features else 1000,
            features['ranked'] if 'ranked' in features else 5000,
            features['popularity'] if 'popularity' in features else 5000,
            features['members'] if 'members' in features else 1000,
            features['year'] if 'year' in features else 2000,
            features['studio_popularity'] if 'studio_popularity' in features else 5.0,
            features['is_movie'],
            features['is_recent'],
            features['is_classic'],
            len(features.get('genres', [])),
        ]).reshape(1, -1)
        
        # Scale features
        try:
            feature_vector_scaled = scaler.transform(feature_vector)
            predicted_score = model.predict(feature_vector_scaled)[0]
            
            # Calculate the difference between predicted and actual
            actual_score = anime.score
            score_difference = predicted_score - actual_score
            
            # Hidden gems: Model predicts higher than actual score
            if score_difference > 0.3:  # Model thinks it should be rated 0.3+ points higher
                hidden_gems.append({
                    'title': anime.title,
                    'actual_score': actual_score,
                    'predicted_score': predicted_score,
                    'difference': score_difference,
                    'episodes': anime.episodes,
                    'year': anime.year,
                    'studios': [s.name for s in anime.studios][:2],
                    'genres': [g.name for g in anime.genres][:3]
                })
        except Exception as e:
            continue  # Skip if feature extraction fails
    
    # Sort by difference (biggest gaps first)
    hidden_gems.sort(key=lambda x: x['difference'], reverse=True)
    
    print(f"\n💎 FOUND {len(hidden_gems)} POTENTIAL HIDDEN GEMS:")
    print("=" * 70)
    print("These anime scored lower than your ML model predicted they should...")
    print()
    
    for i, gem in enumerate(hidden_gems[:15], 1):  # Top 15 hidden gems
        print(f"{i:2d}. {gem['title'][:45]:<45}")
        print(f"    📊 Actual Score:    {gem['actual_score']:.2f}")
        print(f"    🤖 ML Predicted:    {gem['predicted_score']:.2f}")
        print(f"    💎 Underrated by:   {gem['difference']:.2f} points")
        print(f"    📺 Episodes: {gem['episodes']:<4} | Year: {gem['year'] or 'Unknown'}")
        print(f"    � Studios: {', '.join(gem['studios'])}")
        print(f"    🏷️  Genres: {', '.join(gem['genres'])}")
        print()
    
    # Also find overrated anime (for contrast)
    overrated = []
    for anime in all_anime:
        features = ml._extract_anime_features(anime)
        feature_vector = np.array([
            features['episodes'],
            features['score_members'] if 'score_members' in features else 1000,
            features['ranked'] if 'ranked' in features else 5000,
            features['popularity'] if 'popularity' in features else 5000,
            features['members'] if 'members' in features else 1000,
            features['year'] if 'year' in features else 2000,
            features['studio_popularity'] if 'studio_popularity' in features else 5.0,
            features['is_movie'],
            features['is_recent'],
            features['is_classic'],
            len(features.get('genres', [])),
        ]).reshape(1, -1)
        
        try:
            feature_vector_scaled = scaler.transform(feature_vector)
            predicted_score = model.predict(feature_vector_scaled)[0]
            actual_score = anime.score
            score_difference = actual_score - predicted_score
            
            # Overrated: Actual score higher than predicted
            if score_difference > 0.3:
                overrated.append({
                    'title': anime.title,
                    'actual_score': actual_score,
                    'predicted_score': predicted_score,
                    'difference': score_difference
                })
        except:
            continue
    
    overrated.sort(key=lambda x: x['difference'], reverse=True)
    
    print(f"\n📈 FOR COMPARISON - POTENTIALLY OVERRATED ANIME:")
    print("=" * 55)
    print("These scored higher than the model predicted...")
    
    for i, anime in enumerate(overrated[:5], 1):
        print(f"{i}. {anime['title'][:40]:<40} | Actual: {anime['actual_score']:.2f} | Predicted: {anime['predicted_score']:.2f} | +{anime['difference']:.2f}")
    
    print(f"\n🎯 HOW TO USE THESE RESULTS:")
    print("-" * 40)
    print("✅ HIDDEN GEMS: Watch these - they might be better than their scores suggest!")
    print("⚠️  OVERRATED: These might not live up to their high ratings")
    print("🤖 ML INSIGHT: The model learned what typically makes anime highly rated")
    print("💡 DISCOVERY: Find anime you might have overlooked based on low scores")
    
    session.close()
    ml.close()

def analyze_specific_anime():
    """Analyze specific anime to see what the model thinks"""
    
    print("\n" + "="*60)
    print("🔍 ANALYZE SPECIFIC ANIME")
    print("="*60)
    
    session = get_session()
    
    # Get some interesting anime to analyze
    test_cases = [
        "Death Note",
        "Attack on Titan", 
        "One Piece",
        "Naruto",
        "Dragon Ball"
    ]
    
    for title in test_cases:
        anime = session.query(Anime).filter(Anime.title.ilike(f"%{title}%")).first()
        if anime and anime.score:
            print(f"📺 {anime.title}")
            print(f"   Score: {anime.score} | Episodes: {anime.episodes}")
            print(f"   Genres: {[g.name for g in anime.genres][:3]}")
            print(f"   Studios: {[s.name for s in anime.studios]}")
            print()
    
    session.close()

if __name__ == "__main__":
    find_hidden_gems()
    analyze_specific_anime()