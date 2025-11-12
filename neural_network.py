"""
Neural Network for Anime Database
- Rating prediction based on anime features
- Genre classification from synopsis
- Simple recommendation system
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from database import get_session
from models import Anime, Genre
from sqlalchemy import func


class AnimeNeuralNetwork:
    """Neural network for anime-related predictions"""
    
    def __init__(self):
        self.rating_model = None
        self.scaler = StandardScaler()
        self.session = get_session()
        
    def prepare_rating_features(self):
        """Prepare features for rating prediction"""
        
        # Query anime data with features
        query = self.session.query(
            Anime.mal_id,
            Anime.score,
            Anime.episodes,
            Anime.year,
            Anime.popularity,
            Anime.rank,
            func.count(func.distinct('anime_genres.genre_id')).label('genre_count')
        ).outerjoin(
            'genres'
        ).filter(
            Anime.score.isnot(None),
            Anime.episodes.isnot(None),
            Anime.year.isnot(None)
        ).group_by(Anime.mal_id).all()
        
        # Convert to DataFrame
        df = pd.DataFrame(query, columns=[
            'mal_id', 'score', 'episodes', 'year', 
            'popularity', 'rank', 'genre_count'
        ])
        
        # Feature engineering
        df['episodes_log'] = np.log1p(df['episodes'])  # Log transform
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['popularity_rank'] = df['popularity'].rank()
        
        # Select features
        features = [
            'episodes_log', 'year_normalized', 'popularity_rank', 
            'genre_count', 'rank'
        ]
        
        X = df[features].fillna(0)
        y = df['score']
        
        return X, y, df
    
    def build_rating_model(self, input_dim):
        """Build neural network for rating prediction"""
        
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')  # Rating output
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_rating_predictor(self, test_size=0.2, epochs=100):
        """Train neural network to predict anime ratings"""
        
        print("🧠 Preparing data for neural network training...")
        X, y, df = self.prepare_rating_features()
        
        if len(X) < 50:
            return None, "Not enough data for training (need at least 50 anime with scores)"
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42
        )
        
        # Build and train model
        print(f"🏗️ Building neural network with {X.shape[1]} features...")
        self.rating_model = self.build_rating_model(X.shape[1])
        
        print("🚀 Training neural network...")
        history = self.rating_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluate
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        predictions = self.rating_model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        results = {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'test_rmse': rmse,
            'features_used': list(X.columns),
            'history': history.history
        }
        
        print(f"✅ Training complete! Test RMSE: {rmse:.3f}")
        return results, None
    
    def predict_anime_rating(self, anime_id):
        """Predict rating for a specific anime"""
        
        if self.rating_model is None:
            return None, "Model not trained yet"
        
        # Get anime features
        anime = self.session.query(Anime).filter(Anime.mal_id == anime_id).first()
        if not anime:
            return None, "Anime not found"
        
        # Count genres
        genre_count = self.session.query(func.count()).select_from(
            self.session.query(Anime).filter(Anime.mal_id == anime_id).join('genres').subquery()
        ).scalar() or 0
        
        # Prepare features (same as training)
        features = {
            'episodes_log': np.log1p(anime.episodes or 12),
            'year_normalized': ((anime.year or 2020) - 1960) / (2025 - 1960),  # Normalize to training range
            'popularity_rank': anime.popularity or 5000,
            'genre_count': genre_count,
            'rank': anime.rank or 10000
        }
        
        # Convert to array and scale
        feature_array = np.array(list(features.values())).reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_array)
        
        # Predict
        prediction = self.rating_model.predict(feature_scaled, verbose=0)[0][0]
        
        return prediction, features
    
    def get_model_summary(self):
        """Get model information"""
        
        if self.rating_model is None:
            return "No model trained yet"
        
        return {
            'model_type': 'Rating Prediction Neural Network',
            'layers': len(self.rating_model.layers),
            'parameters': self.rating_model.count_params(),
            'architecture': [layer.output_shape for layer in self.rating_model.layers]
        }
    
    def close(self):
        """Close database session"""
        self.session.close()


# Example usage functions
def create_simple_anime_classifier():
    """Create a simple genre classification neural network"""
    
    # This would classify anime into genres based on synopsis
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=10000, output_dim=100),
        keras.layers.LSTM(64, dropout=0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='sigmoid')  # Multi-label output
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def demonstrate_neural_network():
    """Demonstrate neural network capabilities"""
    
    nn = AnimeNeuralNetwork()
    
    try:
        # Train the model
        results, error = nn.train_rating_predictor(epochs=50)
        
        if error:
            print(f"❌ Error: {error}")
            return
        
        print("\n🎯 Neural Network Results:")
        print(f"Training samples: {results['training_samples']}")
        print(f"Test RMSE: {results['test_rmse']:.3f}")
        print(f"Features: {results['features_used']}")
        
        # Test prediction on a random anime
        session = get_session()
        random_anime = session.query(Anime.mal_id, Anime.title).filter(
            Anime.score.isnot(None)
        ).first()
        
        if random_anime:
            prediction, features = nn.predict_anime_rating(random_anime.mal_id)
            actual_score = session.query(Anime.score).filter(
                Anime.mal_id == random_anime.mal_id
            ).scalar()
            
            print(f"\n🎬 Prediction Test:")
            print(f"Anime: {random_anime.title}")
            print(f"Predicted Score: {prediction:.2f}")
            print(f"Actual Score: {actual_score}")
            print(f"Difference: {abs(prediction - actual_score):.2f}")
        
        session.close()
        
    finally:
        nn.close()


if __name__ == "__main__":
    demonstrate_neural_network()