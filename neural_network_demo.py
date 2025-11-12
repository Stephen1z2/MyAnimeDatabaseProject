"""
Simple Neural Network Demo for Anime Recommendations
- Lightweight demo without TensorFlow dependency
- Shows concept of neural network recommendations
- Can be upgraded to full TensorFlow implementation
"""

import numpy as np
import pandas as pd
from database import get_session
from models import Anime, Genre, Studio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SimpleNeuralNetworkDemo:
    """Simplified neural network demo using mathematical concepts"""
    
    def __init__(self):
        self.session = get_session()
        self.anime_features = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.weights = None
        
    def extract_features(self):
        """Extract numerical features from anime data"""
        
        print("Extracting features from anime database...")
        
        # Get anime with complete data
        anime_list = self.session.query(Anime).filter(
            Anime.score.isnot(None),
            Anime.episodes.isnot(None)
        ).all()
        
        features_data = []
        
        for anime in anime_list:
            # Get genres
            genres = [g.name for g in anime.genres]
            
            # Get studios
            studios = [s.name for s in anime.studios]
            
            # Create feature vector
            feature_dict = {
                'mal_id': anime.mal_id,
                'title': anime.title,
                'score': anime.score,
                'episodes': anime.episodes or 12,
                'year': anime.year or 2020,
                'popularity': anime.popularity or 5000,
                'rank': anime.rank or 10000,
                'genre_count': len(genres),
                'has_action': 1 if 'Action' in genres else 0,
                'has_comedy': 1 if 'Comedy' in genres else 0,
                'has_drama': 1 if 'Drama' in genres else 0,
                'has_romance': 1 if 'Romance' in genres else 0,
                'has_fantasy': 1 if 'Fantasy' in genres else 0,
                'has_scifi': 1 if 'Sci-Fi' in genres else 0,
                'has_slice_of_life': 1 if 'Slice of Life' in genres else 0,
                'has_shounen': 1 if 'Shounen' in genres else 0,
                'synopsis_length': len(anime.synopsis) if anime.synopsis else 0,
            }
            
            features_data.append(feature_dict)
        
        self.anime_features = pd.DataFrame(features_data)
        
        # Feature engineering
        self.anime_features['episodes_log'] = np.log1p(self.anime_features['episodes'])
        self.anime_features['year_norm'] = (self.anime_features['year'] - 1960) / (2025 - 1960)
        self.anime_features['popularity_score'] = 1 / (1 + self.anime_features['popularity'] / 1000)
        self.anime_features['rank_score'] = 1 / (1 + self.anime_features['rank'] / 1000)
        self.anime_features['synopsis_norm'] = np.log1p(self.anime_features['synopsis_length']) / 10
        
        print(f"Extracted features for {len(self.anime_features)} anime")
        return self.anime_features
    
    def prepare_feature_matrix(self):
        """Prepare numerical feature matrix for neural network"""
        
        if self.anime_features is None:
            self.extract_features()
        
        # Select numerical features for the "neural network"
        feature_columns = [
            'episodes_log', 'year_norm', 'popularity_score', 'rank_score', 
            'synopsis_norm', 'genre_count', 'has_action', 'has_comedy', 
            'has_drama', 'has_romance', 'has_fantasy', 'has_scifi', 
            'has_slice_of_life', 'has_shounen'
        ]
        
        # Create feature matrix
        self.feature_matrix = self.anime_features[feature_columns].values
        
        # Scale features (like neural network normalization)
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def simulate_neural_network_training(self):
        """Simulate neural network training process"""
        
        print("Simulating neural network training...")
        
        if self.feature_matrix is None:
            self.prepare_feature_matrix()
        
        num_features = self.feature_matrix.shape[1]
        
        # Simulate neural network weights (normally learned through backpropagation)
        # Layer 1: input -> hidden (64 neurons)
        self.weights = {
            'W1': np.random.normal(0, 0.1, (num_features, 64)),
            'b1': np.zeros((1, 64)),
            'W2': np.random.normal(0, 0.1, (64, 32)),
            'b2': np.zeros((1, 32)),
            'W3': np.random.normal(0, 0.1, (32, 16)),
            'b3': np.zeros((1, 16)),
            'W4': np.random.normal(0, 0.1, (16, 1)),
            'b4': np.zeros((1, 1))
        }
        
        # Simulate training iterations
        print("Training neural network layers:")
        print("Layer 1: 14 -> 64 neurons (ReLU activation)")
        print("Layer 2: 64 -> 32 neurons (ReLU activation)")  
        print("Layer 3: 32 -> 16 neurons (ReLU activation)")
        print("Layer 4: 16 -> 1 neuron (Sigmoid output)")
        
        # Calculate some training metrics
        total_params = sum(w.size for w in self.weights.values())
        print(f"Total parameters: {total_params:,}")
        
        return True
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_pass(self, features):
        """Simulate neural network forward pass"""
        
        # Layer 1
        z1 = np.dot(features, self.weights['W1']) + self.weights['b1']
        a1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(a1, self.weights['W2']) + self.weights['b2']
        a2 = self.relu(z2)
        
        # Layer 3
        z3 = np.dot(a2, self.weights['W3']) + self.weights['b3']
        a3 = self.relu(z3)
        
        # Output layer
        z4 = np.dot(a3, self.weights['W4']) + self.weights['b4']
        output = self.sigmoid(z4)
        
        return output[0][0]
    
    def get_neural_recommendations(self, anime_mal_id, top_k=8):
        """Get recommendations using simulated neural network"""
        
        if self.weights is None:
            self.simulate_neural_network_training()
        
        # Find target anime
        target_idx = None
        for i, mal_id in enumerate(self.anime_features['mal_id']):
            if mal_id == anime_mal_id:
                target_idx = i
                break
        
        if target_idx is None:
            return None, "Anime not found in training data"
        
        target_features = self.feature_matrix[target_idx:target_idx+1]
        target_anime = self.anime_features.iloc[target_idx]
        
        print(f"Getting neural network recommendations for: {target_anime['title']}")
        
        # Calculate neural network similarity scores
        similarities = []
        
        for i, anime_row in self.anime_features.iterrows():
            if i == target_idx:
                continue
                
            candidate_features = self.feature_matrix[i:i+1]
            
            # Combine features for similarity calculation
            combined_features = np.concatenate([
                target_features, 
                candidate_features, 
                np.abs(target_features - candidate_features)
            ], axis=1)
            
            # Get similarity score from "neural network"
            similarity_score = self.forward_pass(combined_features)
            
            # Also use traditional cosine similarity for comparison
            cosine_sim = cosine_similarity(target_features, candidate_features)[0][0]
            
            # Combine neural network output with cosine similarity
            final_score = (similarity_score * 0.7) + (cosine_sim * 0.3)
            
            similarities.append({
                'mal_id': int(anime_row['mal_id']),
                'title': anime_row['title'],
                'score': float(anime_row['score']),
                'year': int(anime_row['year']),
                'neural_similarity': float(similarity_score),
                'cosine_similarity': float(cosine_sim),
                'combined_score': float(final_score),
                'episodes': int(anime_row['episodes']),
                'genre_count': int(anime_row['genre_count'])
            })
        
        # Sort by combined score
        similarities.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return similarities[:top_k], None
    
    def get_stats(self):
        """Get statistics about the neural network demo"""
        
        if self.anime_features is None:
            return {}
        
        return {
            'total_anime': len(self.anime_features),
            'feature_dimensions': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
            'avg_score': self.anime_features['score'].mean(),
            'score_range': (self.anime_features['score'].min(), self.anime_features['score'].max()),
            'year_range': (self.anime_features['year'].min(), self.anime_features['year'].max()),
            'avg_episodes': self.anime_features['episodes'].mean()
        }
    
    def close(self):
        """Close database session"""
        self.session.close()


def demo_neural_network():
    """Demonstrate the neural network recommendation system"""
    
    print("Neural Network Anime Recommendation Demo")
    print("=" * 50)
    
    nn_demo = SimpleNeuralNetworkDemo()
    
    try:
        # Prepare data
        features = nn_demo.extract_features()
        nn_demo.prepare_feature_matrix()
        
        # Train the network
        nn_demo.simulate_neural_network_training()
        
        # Get stats
        stats = nn_demo.get_stats()
        print(f"\nDataset Statistics:")
        print(f"Total anime: {stats['total_anime']}")
        print(f"Features: {stats['feature_dimensions']}")
        print(f"Score range: {stats['score_range'][0]:.1f} - {stats['score_range'][1]:.1f}")
        print(f"Average episodes: {stats['avg_episodes']:.1f}")
        
        # Test recommendation
        sample_anime = features.sample(1).iloc[0]
        mal_id = sample_anime['mal_id']
        title = sample_anime['title']
        
        print(f"\nGetting recommendations for: {title}")
        
        recommendations, error = nn_demo.get_neural_recommendations(mal_id, top_k=5)
        
        if error:
            print(f"Error: {error}")
        else:
            print(f"\nTop 5 Neural Network Recommendations:")
            print("-" * 40)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']}")
                print(f"   Score: {rec['score']:.1f} | Year: {rec['year']}")
                print(f"   Neural Similarity: {rec['neural_similarity']:.3f}")
                print(f"   Combined Score: {rec['combined_score']:.3f}")
                print()
    
    finally:
        nn_demo.close()


if __name__ == "__main__":
    demo_neural_network()