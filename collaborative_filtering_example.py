"""
Collaborative Filtering Example with External Data
Advanced recommendation techniques using user-item interactions
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import sqlite3
import requests
import json


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender using external user data
    """
    
    def __init__(self):
        self.user_item_matrix = None
        self.anime_features = None
        self.similarity_matrix = None
        self.svd_model = None
    
    def fetch_user_ratings_sample(self):
        """
        Simulate fetching user rating data from external sources
        
        In practice, you would:
        1. Use MyAnimeList API to get user lists (if available)
        2. Scrape public user profiles (respecting terms of service)  
        3. Use anime database APIs that provide user ratings
        4. Partner with other anime platforms for data sharing
        """
        
        print("👥 Creating synthetic user rating data for demonstration...")
        
        # This is a simplified example - you'd replace this with real user data
        anime_ids = list(range(1, 501))  # Assume 500 anime
        user_ids = list(range(1, 1001))  # Assume 1000 users
        
        # Generate realistic rating patterns
        np.random.seed(42)
        ratings_data = []
        
        # Simulate user preferences
        user_preferences = {}
        genres = ['Action', 'Romance', 'Comedy', 'Drama', 'Fantasy', 'Sci-Fi']
        
        for user_id in user_ids[:100]:  # Sample 100 users for demo
            # Each user has preferred genres
            preferred_genres = np.random.choice(genres, size=np.random.randint(1, 4), replace=False)
            user_preferences[user_id] = preferred_genres
            
            # Each user rates 20-100 anime
            num_ratings = np.random.randint(20, 101)
            rated_anime = np.random.choice(anime_ids, size=num_ratings, replace=False)
            
            for anime_id in rated_anime:
                # Simulate realistic rating based on "preferences"
                base_rating = np.random.normal(7.5, 1.5)
                
                # Add genre preference bonus
                if np.random.random() < 0.3:  # 30% chance anime matches preference
                    base_rating += np.random.normal(1.0, 0.5)
                
                # Clamp to 1-10 scale
                rating = max(1, min(10, base_rating))
                
                ratings_data.append({
                    'user_id': user_id,
                    'anime_id': anime_id,
                    'rating': rating,
                    'user_genres': preferred_genres.tolist()
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        print(f"✅ Created {len(ratings_df)} synthetic user ratings")
        
        return ratings_df, user_preferences
    
    def create_user_item_matrix(self, ratings_df):
        """Create user-item interaction matrix"""
        
        print("📊 Creating user-item matrix...")
        
        # Pivot ratings into matrix format
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='anime_id', 
            values='rating',
            fill_value=0
        )
        
        print(f"✅ Matrix shape: {self.user_item_matrix.shape}")
        return self.user_item_matrix
    
    def train_collaborative_model(self, n_components=50):
        """Train SVD-based collaborative filtering model"""
        
        if self.user_item_matrix is None:
            print("❌ No user-item matrix available")
            return
        
        print(f"🤖 Training collaborative filtering model with {n_components} components...")
        
        # Apply SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        
        # Fit on user-item matrix
        user_factors = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Get item factors
        item_factors = self.svd_model.components_.T
        
        # Compute item-item similarity
        self.similarity_matrix = cosine_similarity(item_factors)
        
        print(f"✅ Model trained. Explained variance: {self.svd_model.explained_variance_ratio_.sum():.3f}")
        
        return user_factors, item_factors
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations for a specific user"""
        
        if self.user_item_matrix is None or self.svd_model is None:
            return []
        
        # Get user's ratings
        if user_id not in self.user_item_matrix.index:
            print(f"❌ User {user_id} not found")
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Find items user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index
        
        if len(unrated_items) == 0:
            return []
        
        # Get user factors
        user_vector = self.svd_model.transform(user_ratings.values.reshape(1, -1))[0]
        
        # Get item factors for unrated items
        item_indices = [list(self.user_item_matrix.columns).index(item) for item in unrated_items]
        item_factors = self.svd_model.components_[:, item_indices].T
        
        # Compute predicted ratings
        predicted_ratings = np.dot(user_vector, item_factors.T)
        
        # Get top recommendations
        top_indices = np.argsort(predicted_ratings)[-n_recommendations:][::-1]
        top_items = [unrated_items[i] for i in top_indices]
        top_scores = [predicted_ratings[i] for i in top_indices]
        
        recommendations = []
        for item_id, score in zip(top_items, top_scores):
            recommendations.append({
                'anime_id': item_id,
                'predicted_rating': score,
                'confidence': min(1.0, score / 10.0)  # Normalize confidence
            })
        
        return recommendations
    
    def get_similar_items(self, anime_id, n_similar=10):
        """Find similar anime using collaborative filtering"""
        
        if self.similarity_matrix is None:
            return []
        
        anime_list = list(self.user_item_matrix.columns)
        
        if anime_id not in anime_list:
            return []
        
        anime_index = anime_list.index(anime_id)
        similarities = self.similarity_matrix[anime_index]
        
        # Get most similar items (excluding self)
        similar_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        similar_items = []
        for idx in similar_indices:
            similar_items.append({
                'anime_id': anime_list[idx],
                'similarity_score': similarities[idx]
            })
        
        return similar_items


def demonstrate_collaborative_filtering():
    """Demonstrate collaborative filtering with synthetic data"""
    
    print("🤝 Collaborative Filtering Recommendation Demo")
    print("=" * 60)
    
    # Initialize recommender
    recommender = CollaborativeFilteringRecommender()
    
    # Step 1: Get user rating data (synthetic for demo)
    ratings_df, user_prefs = recommender.fetch_user_ratings_sample()
    
    # Step 2: Create user-item matrix
    recommender.create_user_item_matrix(ratings_df)
    
    # Step 3: Train collaborative model
    recommender.train_collaborative_model(n_components=30)
    
    # Step 4: Get recommendations for sample users
    print("\n🎯 Sample Recommendations:")
    print("-" * 40)
    
    sample_users = [1, 2, 3, 4, 5]
    
    for user_id in sample_users:
        print(f"\n👤 User {user_id} (Preferences: {user_prefs.get(user_id, 'Unknown')}):")
        
        recommendations = recommender.get_recommendations(user_id, n_recommendations=5)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. Anime {rec['anime_id']:3d} - Score: {rec['predicted_rating']:.2f} (Confidence: {rec['confidence']:.2f})")
    
    # Step 5: Show item similarity
    print("\n🔍 Item Similarity Example:")
    print("-" * 30)
    
    sample_anime = 100
    similar_items = recommender.get_similar_items(sample_anime, n_similar=5)
    
    print(f"\nAnime similar to Anime {sample_anime}:")
    for i, item in enumerate(similar_items, 1):
        print(f"   {i}. Anime {item['anime_id']:3d} - Similarity: {item['similarity_score']:.3f}")
    
    print("\n✅ Collaborative filtering demonstration complete!")
    print("\n💡 To implement with real data:")
    print("   1. Collect user rating data from MAL API or other sources")
    print("   2. Build user profiles based on rating history") 
    print("   3. Use matrix factorization for scalable recommendations")
    print("   4. Combine with content-based features for hybrid approach")


class HybridRecommendationSystem:
    """
    Hybrid system combining collaborative filtering with content-based features
    """
    
    def __init__(self):
        self.cf_recommender = CollaborativeFilteringRecommender()
        self.content_features = None
        self.content_similarity = None
    
    def create_content_features(self, anime_df):
        """Create content-based features from anime metadata"""
        
        print("📝 Creating content-based features...")
        
        # Combine text features
        anime_df['combined_features'] = (
            anime_df['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
            anime_df['studios'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '') + ' ' +
            anime_df['synopsis'].fillna('') + ' ' +
            anime_df['type'].fillna('') + ' ' +
            anime_df['source_material'].fillna('')
        )
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.content_features = tfidf.fit_transform(anime_df['combined_features'])
        self.content_similarity = cosine_similarity(self.content_features)
        
        print(f"✅ Content features created: {self.content_features.shape}")
        
        return self.content_features
    
    def get_hybrid_recommendations(self, user_id, anime_df, n_recommendations=10, cf_weight=0.7):
        """
        Get hybrid recommendations combining collaborative and content-based filtering
        
        Args:
            user_id: Target user ID
            anime_df: DataFrame with anime metadata
            n_recommendations: Number of recommendations to return
            cf_weight: Weight for collaborative filtering (0-1, remainder goes to content)
        """
        
        # Get collaborative filtering recommendations
        cf_recs = self.cf_recommender.get_recommendations(user_id, n_recommendations * 2)
        
        if not cf_recs:
            print(f"❌ No CF recommendations for user {user_id}")
            return []
        
        # Get content-based scores for CF recommendations
        if self.content_similarity is not None:
            # Get user's rating history to understand preferences
            user_ratings = self.cf_recommender.user_item_matrix.loc[user_id]
            liked_anime = user_ratings[user_ratings >= 8.0].index.tolist()
            
            # Calculate content similarity scores for each CF recommendation
            for rec in cf_recs:
                anime_id = rec['anime_id']
                
                if anime_id < len(self.content_similarity):
                    # Find average similarity to user's liked anime
                    content_scores = []
                    
                    for liked_id in liked_anime[:10]:  # Limit to top 10 liked
                        if liked_id < len(self.content_similarity):
                            content_scores.append(self.content_similarity[anime_id][liked_id])
                    
                    avg_content_score = np.mean(content_scores) if content_scores else 0.5
                    
                    # Combine CF and content scores
                    hybrid_score = (cf_weight * rec['predicted_rating'] + 
                                  (1 - cf_weight) * avg_content_score * 10)
                    
                    rec['content_score'] = avg_content_score
                    rec['hybrid_score'] = hybrid_score
                else:
                    rec['content_score'] = 0.5
                    rec['hybrid_score'] = rec['predicted_rating']
        
        # Sort by hybrid score and return top recommendations
        cf_recs.sort(key=lambda x: x.get('hybrid_score', x['predicted_rating']), reverse=True)
        
        return cf_recs[:n_recommendations]


def main():
    """Main function to demonstrate enhanced ML recommendations"""
    
    print("🚀 Enhanced Anime ML Recommendations with External Data")
    print("=" * 70)
    
    print("\n📋 This demo shows how to:")
    print("1. 👥 Use collaborative filtering with user rating data")
    print("2. 📝 Combine with content-based features") 
    print("3. 🔗 Create hybrid recommendation systems")
    print("4. 🌐 Integrate external data sources")
    
    # Run collaborative filtering demo
    demonstrate_collaborative_filtering()
    
    print("\n" + "=" * 70)
    print("🎯 Key Strategies for Better Recommendations:")
    print()
    print("1. 🌐 **External Data Sources:**")
    print("   • MyAnimeList user ratings via API")
    print("   • AniList user preferences and lists")
    print("   • Seasonal popularity trends")
    print("   • Studio performance metrics")
    print()
    print("2. 🤖 **Advanced ML Techniques:**")
    print("   • Matrix factorization (SVD, NMF)")
    print("   • Neural collaborative filtering")
    print("   • Deep learning embeddings")
    print("   • Ensemble methods")
    print()
    print("3. 📊 **Feature Engineering:**")
    print("   • User-item interactions")
    print("   • Temporal patterns (viewing time)")
    print("   • Social features (friend preferences)")
    print("   • Content similarity metrics")
    print()
    print("4. 🎯 **Evaluation Methods:**")
    print("   • A/B testing with users")
    print("   • Cross-validation on historical data")
    print("   • Diversity and novelty metrics")
    print("   • Cold start problem solutions")
    
    print("\n✨ Next steps: Run the external_data_collector.py script")
    print("   to gather real anime data for training!")


if __name__ == "__main__":
    main()