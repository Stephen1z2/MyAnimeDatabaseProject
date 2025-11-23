"""
Machine Learning Module for Anime Database
- Score prediction using regression algorithms
- Genre classification using text analysis
- Clustering for anime discovery
- Feature engineering and model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from database import get_session
from models import Anime, Genre, Studio, anime_genres
from sqlalchemy import func
import warnings
warnings.filterwarnings('ignore')


class AnimeMLAnalyzer:
    """Comprehensive Machine Learning Analysis for Anime Database"""
    
    def __init__(self):
        self.session = get_session()
        self.data = None
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load anime data and engineer features for ML"""
        
        print("🔄 Loading and preparing anime data for machine learning...")
        
        # Query anime data with all features
        anime_list = self.session.query(Anime).filter(
            Anime.score.isnot(None),
            Anime.episodes.isnot(None)
        ).all()
        
        data_rows = []
        
        for anime in anime_list:
            # Get genres
            genres = [g.name for g in anime.genres]
            
            # Get studios
            studios = [s.name for s in anime.studios]
            studio_name = studios[0] if studios else 'Unknown'
            
            # Feature engineering
            row = {
                # Basic info
                'mal_id': anime.mal_id,
                'title': anime.title,
                'score': anime.score,
                'episodes': anime.episodes or 12,
                'year': anime.year or 2020,
                'popularity': anime.popularity or 5000,
                'rank': anime.rank or 10000,
                'type': anime.type or 'TV',
                'status': anime.status or 'Finished',
                'studio': studio_name,
                'synopsis': anime.synopsis or '',
                
                # Engineered features
                'episodes_log': np.log1p(anime.episodes or 12),
                'year_since_2000': max(0, (anime.year or 2020) - 2000),
                'popularity_score': 1 / (1 + (anime.popularity or 5000) / 1000),
                'rank_score': 1 / (1 + (anime.rank or 10000) / 1000),
                'synopsis_length': len(anime.synopsis) if anime.synopsis else 0,
                'synopsis_words': len(anime.synopsis.split()) if anime.synopsis else 0,
                'genre_count': len(genres),
                
                # Genre features (one-hot encoding)
                'has_action': 1 if 'Action' in genres else 0,
                'has_adventure': 1 if 'Adventure' in genres else 0,
                'has_comedy': 1 if 'Comedy' in genres else 0,
                'has_drama': 1 if 'Drama' in genres else 0,
                'has_fantasy': 1 if 'Fantasy' in genres else 0,
                'has_romance': 1 if 'Romance' in genres else 0,
                'has_scifi': 1 if 'Sci-Fi' in genres else 0,
                'has_slice_of_life': 1 if 'Slice of Life' in genres else 0,
                'has_shounen': 1 if 'Shounen' in genres else 0,
                'has_seinen': 1 if 'Seinen' in genres else 0,
                'has_supernatural': 1 if 'Supernatural' in genres else 0,
                'has_thriller': 1 if 'Thriller' in genres else 0,
                
                # Studio popularity (based on anime count)
                'studio_anime_count': self._get_studio_popularity(studio_name),
                
                # Decade categorization
                'decade': (anime.year // 10) * 10 if anime.year else 2020,
                'is_recent': 1 if (anime.year or 2020) >= 2010 else 0,
                'is_classic': 1 if (anime.year or 2020) < 1990 else 0,
            }
            
            data_rows.append(row)
        
        self.data = pd.DataFrame(data_rows)
        
        # Additional feature engineering
        self.data['episodes_category'] = pd.cut(
            self.data['episodes'], 
            bins=[0, 1, 12, 26, 50, float('inf')], 
            labels=['Movie', 'Short', 'Standard', 'Long', 'Very_Long']
        )
        
        # Studio tier (based on average score)
        studio_scores = self.data.groupby('studio')['score'].mean()
        self.data['studio_tier'] = self.data['studio'].map(
            lambda x: 'Top' if studio_scores.get(x, 0) > 8.2 else 
                     'Good' if studio_scores.get(x, 0) > 8.0 else 'Standard'
        )
        
        print(f"✅ Prepared {len(self.data)} anime records with {len(self.data.columns)} features")
        return self.data
    
    def _get_studio_popularity(self, studio_name):
        """Get studio popularity based on number of anime produced"""
        try:
            count = self.session.query(func.count(Anime.id)).join(
                Anime.studios
            ).filter(Studio.name == studio_name).scalar() or 1
            return min(count, 50)  # Cap at 50 for normalization
        except:
            return 1
    
    def train_score_predictor(self, test_size=0.2):
        """Train machine learning models to predict anime scores"""
        
        if self.data is None:
            self.load_and_prepare_data()
        
        print("🤖 Training score prediction models...")
        
        # Select features for prediction
        feature_columns = [
            'episodes_log', 'year_since_2000', 'popularity_score', 'rank_score',
            'synopsis_length', 'synopsis_words', 'genre_count', 'studio_anime_count',
            'has_action', 'has_adventure', 'has_comedy', 'has_drama',
            'has_fantasy', 'has_romance', 'has_scifi', 'has_slice_of_life',
            'has_shounen', 'has_seinen', 'has_supernatural', 'has_thriller',
            'is_recent', 'is_classic'
        ]
        
        X = self.data[feature_columns]
        y = self.data['score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['score_predictor'] = scaler
        
        # Train multiple models
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Random Forest Tuned': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
            )
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"   Training {name}...")
            
            if 'Linear' in name:
                # Use scaled features for linear regression
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                # Store scaled version for linear models
                self.models[name] = (model, scaler)
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.models[name] = model
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if 'Linear' in name:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'actual': y_test
            }
            
            print(f"      RMSE: {rmse:.4f}, R²: {r2:.4f}, CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Feature importance (from Random Forest)
        if 'Random Forest' in results:
            rf_model = self.models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = feature_importance
            
            print("\n🎯 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        self.results['score_prediction'] = results
        return results
    
    def predict_anime_score(self, anime_features, model_name='Random Forest'):
        """Predict score for new anime based on features"""
        
        if model_name not in self.models:
            return None, f"Model {model_name} not trained yet"
        
        model = self.models[model_name]
        
        # Handle different model types
        if isinstance(model, tuple):  # Linear models with scaler
            model_obj, scaler = model
            features_scaled = scaler.transform([anime_features])
            prediction = model_obj.predict(features_scaled)[0]
        else:
            prediction = model.predict([anime_features])[0]
        
        return prediction, None
    
    def train_genre_classifier(self):
        """Train model to predict genres from synopsis"""
        
        if self.data is None:
            self.load_and_prepare_data()
        
        print("📝 Training genre classification model...")
        
        # Prepare text data
        text_data = self.data['synopsis'].fillna('')
        
        # Create genre matrix
        genre_columns = [
            'has_action', 'has_adventure', 'has_comedy', 'has_drama',
            'has_fantasy', 'has_romance', 'has_scifi', 'has_slice_of_life',
            'has_shounen', 'has_seinen'
        ]
        
        y_genres = self.data[genre_columns]
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        X_tfidf = tfidf.fit_transform(text_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_genres, test_size=0.2, random_state=42
        )
        
        # Train multi-output classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        multi_classifier = MultiOutputClassifier(rf_classifier)
        
        multi_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = multi_classifier.predict(X_test)
        
        # Calculate accuracy for each genre
        genre_accuracies = {}
        for i, genre in enumerate(genre_columns):
            accuracy = (y_pred[:, i] == y_test.iloc[:, i]).mean()
            genre_accuracies[genre] = accuracy
        
        self.models['genre_classifier'] = (multi_classifier, tfidf)
        self.results['genre_classification'] = {
            'accuracies': genre_accuracies,
            'overall_accuracy': np.mean(list(genre_accuracies.values()))
        }
        
        print(f"✅ Genre classification trained. Average accuracy: {np.mean(list(genre_accuracies.values())):.3f}")
        
        return self.results['genre_classification']
    
    def perform_anime_clustering(self, n_clusters=8):
        """Cluster anime for discovery and analysis"""
        
        if self.data is None:
            self.load_and_prepare_data()
        
        print(f"🔍 Performing K-means clustering with {n_clusters} clusters...")
        
        # Select features for clustering
        cluster_features = [
            'episodes_log', 'year_since_2000', 'genre_count',
            'has_action', 'has_comedy', 'has_drama', 'has_fantasy',
            'has_romance', 'has_scifi', 'is_recent'
        ]
        
        X_cluster = self.data[cluster_features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to data
        self.data['cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            
            cluster_analysis[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_score': cluster_data['score'].mean(),
                'avg_episodes': cluster_data['episodes'].mean(),
                'avg_year': cluster_data['year'].mean(),
                'top_genres': self._get_cluster_top_genres(cluster_data),
                'sample_anime': cluster_data['title'].head(3).tolist()
            }
        
        self.models['clustering'] = (kmeans, scaler)
        self.results['clustering'] = cluster_analysis
        
        print("✅ Clustering completed. Cluster analysis:")
        for cluster, info in cluster_analysis.items():
            print(f"   {cluster}: {info['size']} anime, avg score: {info['avg_score']:.2f}")
        
        return cluster_analysis
    
    def _get_cluster_top_genres(self, cluster_data):
        """Get top genres for a cluster"""
        genre_cols = [col for col in cluster_data.columns if col.startswith('has_')]
        genre_sums = cluster_data[genre_cols].sum().sort_values(ascending=False)
        return genre_sums.head(3).to_dict()
    
    def generate_ml_report(self):
        """Generate comprehensive ML analysis report"""
        
        if not self.results:
            print("❌ No ML results available. Run training first.")
            return None
        
        report = {
            'dataset_info': {
                'total_anime': len(self.data),
                'features_count': len([col for col in self.data.columns if col.startswith(('has_', 'episodes_', 'year_'))]),
                'score_range': (self.data['score'].min(), self.data['score'].max()),
                'year_range': (self.data['year'].min(), self.data['year'].max())
            },
            'model_performance': self.results,
            'best_model': self._get_best_model(),
            'recommendations': self._generate_ml_recommendations()
        }
        
        return report
    
    def _get_best_model(self):
        """Determine best performing model"""
        if 'score_prediction' not in self.results:
            return None
        
        models = self.results['score_prediction']
        best_model = max(
            [(name, info) for name, info in models.items() if isinstance(info, dict) and 'r2' in info],
            key=lambda x: x[1]['r2']
        )
        
        return best_model[0] if best_model else None
    
    def _generate_ml_recommendations(self):
        """Generate recommendations for improving ML models"""
        recommendations = []
        
        if 'score_prediction' in self.results:
            best_r2 = max(
                [info['r2'] for name, info in self.results['score_prediction'].items() 
                 if isinstance(info, dict) and 'r2' in info]
            )
            
            if best_r2 < 0.7:
                recommendations.append("Consider feature engineering: studio ratings, seasonal trends")
            if best_r2 < 0.5:
                recommendations.append("Try deep learning models for better pattern recognition")
            
        recommendations.append("Experiment with ensemble methods combining multiple algorithms")
        recommendations.append("Add more external features: director info, voice actors")
        
        return recommendations
    
    def close(self):
        """Close database session"""
        self.session.close()


def demonstrate_anime_ml():
    """Demonstrate machine learning capabilities"""
    
    print("🤖 ANIME DATABASE MACHINE LEARNING DEMONSTRATION")
    print("=" * 60)
    
    analyzer = AnimeMLAnalyzer()
    
    try:
        # Load data
        data = analyzer.load_and_prepare_data()
        print(f"\n📊 Dataset: {len(data)} anime with {len(data.columns)} features")
        
        # Train score predictor
        print("\n1. SCORE PREDICTION MODELS")
        print("-" * 30)
        score_results = analyzer.train_score_predictor()
        
        # Train genre classifier
        print("\n2. GENRE CLASSIFICATION")
        print("-" * 30)
        genre_results = analyzer.train_genre_classifier()
        
        # Perform clustering
        print("\n3. ANIME CLUSTERING")
        print("-" * 30)
        cluster_results = analyzer.perform_anime_clustering()
        
        # Generate report
        print("\n4. ML ANALYSIS REPORT")
        print("-" * 30)
        report = analyzer.generate_ml_report()
        
        if report:
            print(f"📈 Best Model: {report['best_model']}")
            print("🎯 ML Recommendations:")
            for rec in report['recommendations']:
                print(f"   • {rec}")
        
        print("\n✅ Machine Learning demonstration completed!")
        print("Your anime database is perfect for advanced ML experiments!")
        
    finally:
        analyzer.close()


if __name__ == "__main__":
    demonstrate_anime_ml()