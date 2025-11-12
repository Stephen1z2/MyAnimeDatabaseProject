"""
Neural Network Recommendation System for Anime Database
- Collaborative filtering with neural networks
- Content-based recommendations using embeddings
- Hybrid recommendation system
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from database import get_session
from models import Anime, Genre, Studio, anime_genres, anime_studios
from sqlalchemy import func, text
import warnings
warnings.filterwarnings('ignore')


class AnimeRecommendationNN:
    """Neural Network-based Anime Recommendation System"""
    
    def __init__(self):
        self.session = get_session()
        self.content_model = None
        self.collaborative_model = None
        self.anime_encoder = LabelEncoder()
        self.genre_encoder = LabelEncoder()
        self.studio_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.anime_features = None
        self.trained = False
        
    def prepare_content_features(self):
        """Prepare content-based features for anime"""
        
        print("🎬 Preparing content-based features...")
        
        # Query anime with all features
        anime_data = []
        
        anime_list = self.session.query(Anime).filter(
            Anime.score.isnot(None),
            Anime.episodes.isnot(None)
        ).all()
        
        for anime in anime_list:
            # Get genres
            genres = [g.name for g in anime.genres]
            genre_str = '|'.join(genres) if genres else 'Unknown'
            
            # Get studios
            studios = [s.name for s in anime.studios]
            studio_str = studios[0] if studios else 'Unknown'
            
            anime_data.append({
                'mal_id': anime.mal_id,
                'title': anime.title,
                'score': anime.score,
                'episodes': anime.episodes or 12,
                'year': anime.year or 2020,
                'popularity': anime.popularity or 5000,
                'rank': anime.rank or 10000,
                'type': anime.type or 'TV',
                'status': anime.status or 'Finished',
                'genres': genre_str,
                'studio': studio_str,
                'genre_count': len(genres),
                'synopsis_length': len(anime.synopsis) if anime.synopsis else 0
            })
        
        df = pd.DataFrame(anime_data)
        
        if len(df) == 0:
            return None, "No anime data available"
        
        # Feature engineering
        df['episodes_log'] = np.log1p(df['episodes'])
        df['year_norm'] = (df['year'] - 1960) / (2025 - 1960)
        df['popularity_norm'] = 1 / (1 + df['popularity'] / 1000)  # Inverse popularity
        df['rank_norm'] = 1 / (1 + df['rank'] / 1000)  # Inverse rank
        df['synopsis_norm'] = np.log1p(df['synopsis_length']) / 10
        
        # Encode categorical features
        df['studio_encoded'] = self.studio_encoder.fit_transform(df['studio'])
        df['type_encoded'] = LabelEncoder().fit_transform(df['type'])
        df['status_encoded'] = LabelEncoder().fit_transform(df['status'])
        
        # Create genre features (multi-hot encoding)
        all_genres = set()
        for genre_str in df['genres']:
            all_genres.update(genre_str.split('|'))
        all_genres.discard('Unknown')
        
        # Create binary genre columns
        for genre in sorted(all_genres):
            df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)
        
        self.anime_features = df
        print(f"✅ Prepared features for {len(df)} anime")
        return df, None
    
    def build_content_model(self, num_features):
        """Build content-based neural network model"""
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(num_features,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Output similarity score 0-1
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mae']
        )
        
        return model
    
    def create_training_pairs(self, df):
        """Create training pairs for similarity learning"""
        
        print("🔄 Creating training pairs...")
        
        pairs = []
        labels = []
        
        # Create positive pairs (similar anime)
        for i, anime1 in df.iterrows():
            for j, anime2 in df.iterrows():
                if i >= j:
                    continue
                
                # Calculate similarity based on multiple factors
                score_sim = 1 - abs(anime1['score'] - anime2['score']) / 10  # Score similarity
                genre_sim = len(set(anime1['genres'].split('|')) & set(anime2['genres'].split('|'))) / \
                           len(set(anime1['genres'].split('|')) | set(anime2['genres'].split('|')))  # Jaccard
                studio_sim = 1 if anime1['studio'] == anime2['studio'] else 0
                year_sim = 1 - abs(anime1['year'] - anime2['year']) / 50  # Year proximity
                
                # Combined similarity
                similarity = (score_sim * 0.4 + genre_sim * 0.4 + studio_sim * 0.1 + year_sim * 0.1)
                
                # Create pair features (concatenate or difference)
                feature1 = self.extract_features(anime1)
                feature2 = self.extract_features(anime2)
                pair_features = np.concatenate([feature1, feature2, np.abs(feature1 - feature2)])
                
                pairs.append(pair_features)
                labels.append(1 if similarity > 0.6 else 0)  # Binary similarity threshold
        
        return np.array(pairs), np.array(labels)
    
    def extract_features(self, anime_row):
        """Extract numerical features from anime row"""
        
        features = []
        
        # Numerical features
        features.extend([
            anime_row['episodes_log'],
            anime_row['year_norm'],
            anime_row['popularity_norm'],
            anime_row['rank_norm'],
            anime_row['synopsis_norm'],
            anime_row['genre_count'],
            anime_row['studio_encoded'],
            anime_row['type_encoded'],
            anime_row['status_encoded']
        ])
        
        # Genre features
        genre_cols = [col for col in anime_row.index if col.startswith('genre_')]
        features.extend([anime_row[col] for col in genre_cols])
        
        return np.array(features)
    
    def train_content_model(self, epochs=50):
        """Train the content-based recommendation model"""
        
        df, error = self.prepare_content_features()
        if error:
            return None, error
        
        if len(df) < 20:
            return None, "Need at least 20 anime for training"
        
        # Create training data
        X, y = self.create_training_pairs(df)
        
        if len(X) == 0:
            return None, "No training pairs created"
        
        print(f"🚀 Training with {len(X)} pairs...")
        
        # Build model
        num_features = X.shape[1]
        self.content_model = self.build_content_model(num_features)
        
        # Train
        history = self.content_model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.trained = True
        
        # Evaluate
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        results = {
            'training_pairs': len(X),
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'anime_count': len(df)
        }
        
        print(f"✅ Training complete! Validation accuracy: {val_acc:.3f}")
        return results, None
    
    def get_recommendations(self, anime_mal_id, top_k=10):
        """Get neural network recommendations for an anime"""
        
        if not self.trained or self.content_model is None:
            return None, "Model not trained yet"
        
        # Find target anime
        target_anime = None
        for _, row in self.anime_features.iterrows():
            if row['mal_id'] == anime_mal_id:
                target_anime = row
                break
        
        if target_anime is None:
            return None, "Anime not found in trained data"
        
        print(f"🎯 Getting recommendations for: {target_anime['title']}")
        
        # Calculate similarities with all other anime
        target_features = self.extract_features(target_anime)
        similarities = []
        
        for _, candidate in self.anime_features.iterrows():
            if candidate['mal_id'] == anime_mal_id:
                continue  # Skip self
            
            candidate_features = self.extract_features(candidate)
            
            # Create pair features for the model
            pair_features = np.concatenate([
                target_features, 
                candidate_features, 
                np.abs(target_features - candidate_features)
            ])
            
            # Get similarity score from neural network
            similarity_score = self.content_model.predict(
                pair_features.reshape(1, -1), 
                verbose=0
            )[0][0]
            
            similarities.append({
                'mal_id': int(candidate['mal_id']),
                'title': candidate['title'],
                'score': float(candidate['score']),
                'similarity': float(similarity_score),
                'genres': candidate['genres'],
                'studio': candidate['studio'],
                'year': int(candidate['year'])
            })
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k], None
    
    def create_anime_embeddings(self):
        """Create anime embeddings using autoencoder"""
        
        if self.anime_features is None:
            return None, "No anime features prepared"
        
        # Extract features for all anime
        feature_matrix = []
        for _, anime in self.anime_features.iterrows():
            features = self.extract_features(anime)
            feature_matrix.append(features)
        
        X = np.array(feature_matrix)
        X_scaled = self.scaler.fit_transform(X)
        
        # Build autoencoder for embeddings
        input_dim = X_scaled.shape[1]
        embedding_dim = 32
        
        # Encoder
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = keras.layers.Dense(embedding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        print("🔧 Training autoencoder for embeddings...")
        autoencoder.fit(X_scaled, X_scaled, epochs=50, verbose=0)
        
        # Generate embeddings
        embeddings = encoder.predict(X_scaled, verbose=0)
        
        # Store embeddings with anime IDs
        embedding_dict = {}
        for i, (_, anime) in enumerate(self.anime_features.iterrows()):
            embedding_dict[anime['mal_id']] = embeddings[i]
        
        return embedding_dict, encoder
    
    def get_model_info(self):
        """Get information about the trained models"""
        
        info = {
            'trained': self.trained,
            'anime_count': len(self.anime_features) if self.anime_features is not None else 0,
            'content_model': None,
            'embedding_model': None
        }
        
        if self.content_model:
            info['content_model'] = {
                'layers': len(self.content_model.layers),
                'parameters': self.content_model.count_params(),
                'input_shape': self.content_model.input_shape,
                'output_shape': self.content_model.output_shape
            }
        
        return info
    
    def save_model(self, filepath="anime_recommendation_nn.h5"):
        """Save the trained model"""
        if self.content_model:
            self.content_model.save(filepath)
            return f"Model saved to {filepath}"
        return "No model to save"
    
    def load_model(self, filepath="anime_recommendation_nn.h5"):
        """Load a pre-trained model"""
        try:
            self.content_model = keras.models.load_model(filepath)
            self.trained = True
            return f"Model loaded from {filepath}"
        except:
            return "Failed to load model"
    
    def close(self):
        """Close database connection"""
        self.session.close()


def demonstrate_recommendation_nn():
    """Demonstrate the neural network recommendation system"""
    
    print("🧠 Neural Network Anime Recommendation Demo")
    print("=" * 50)
    
    nn = AnimeRecommendationNN()
    
    try:
        # Train the model
        results, error = nn.train_content_model(epochs=30)
        
        if error:
            print(f"❌ Error: {error}")
            return
        
        print(f"\n📊 Training Results:")
        print(f"Anime count: {results['anime_count']}")
        print(f"Training pairs: {results['training_pairs']}")
        print(f"Final accuracy: {results['final_accuracy']:.3f}")
        print(f"Validation accuracy: {results['val_accuracy']:.3f}")
        
        # Test recommendations
        print(f"\n🎯 Testing Recommendations:")
        
        # Get a random anime for testing
        sample_anime = nn.anime_features.sample(1).iloc[0]
        mal_id = sample_anime['mal_id']
        title = sample_anime['title']
        
        recommendations, error = nn.get_recommendations(mal_id, top_k=5)
        
        if error:
            print(f"❌ Error getting recommendations: {error}")
            return
        
        print(f"\n🎬 Recommendations for: {title}")
        print("-" * 40)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (Score: {rec['score']:.1f})")
            print(f"   Similarity: {rec['similarity']:.3f}")
            print(f"   Year: {rec['year']}, Studio: {rec['studio']}")
            print(f"   Genres: {rec['genres'].replace('|', ', ')}")
            print()
        
        # Model info
        info = nn.get_model_info()
        print(f"📈 Model Info:")
        print(f"Parameters: {info['content_model']['parameters']:,}")
        print(f"Layers: {info['content_model']['layers']}")
        
    finally:
        nn.close()


if __name__ == "__main__":
    demonstrate_recommendation_nn()