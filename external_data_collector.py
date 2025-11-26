"""
External Anime Data Collection Script
Fetches anime data from multiple sources for enhanced ML training

This script helps you collect anime data from:
1. MyAnimeList (via Jikan API) 
2. AniList (GraphQL API)
3. Seasonal anime trends
4. Genre popularity data
"""

import requests
import json
import time
import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime


class AnimeDataCollector:
    """Collect anime data from external sources"""
    
    def __init__(self):
        self.cache_dir = Path("external_data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.collected_data = {}
    
    def fetch_mal_top_anime(self, max_pages=20):
        """
        Fetch top anime from MyAnimeList via Jikan API
        
        Benefits for ML:
        - High-quality ratings from large user base
        - Studio performance data
        - Genre popularity insights
        - Member counts for popularity metrics
        """
        print(f"🔥 Fetching top anime from MyAnimeList...")
        
        all_anime = []
        
        for page in range(1, max_pages + 1):
            try:
                print(f"   Page {page}/{max_pages}...")
                
                url = f"https://api.jikan.moe/v4/top/anime?page={page}&limit=25"
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    anime_list = data.get('data', [])
                    
                    for anime in anime_list:
                        anime_info = {
                            'source': 'MyAnimeList_Top',
                            'mal_id': anime.get('mal_id'),
                            'title': anime.get('title', ''),
                            'title_english': anime.get('title_english', ''),
                            'score': anime.get('score'),
                            'scored_by': anime.get('scored_by', 0),
                            'rank': anime.get('rank'),
                            'popularity': anime.get('popularity'),
                            'members': anime.get('members', 0),
                            'favorites': anime.get('favorites', 0),
                            'synopsis': anime.get('synopsis', ''),
                            'type': anime.get('type', ''),
                            'episodes': anime.get('episodes'),
                            'status': anime.get('status', ''),
                            'aired_from': anime.get('aired', {}).get('from'),
                            'aired_to': anime.get('aired', {}).get('to'),
                            'year': anime.get('year'),
                            'season': anime.get('season', ''),
                            'studios': [studio['name'] for studio in anime.get('studios', [])],
                            'genres': [genre['name'] for genre in anime.get('genres', [])],
                            'themes': [theme['name'] for theme in anime.get('themes', [])],
                            'demographics': [demo['name'] for demo in anime.get('demographics', [])],
                            'rating': anime.get('rating', ''),
                            'duration': anime.get('duration', ''),
                            'source_material': anime.get('source', ''),
                            'producers': [prod['name'] for prod in anime.get('producers', [])],
                            'licensors': [lic['name'] for lic in anime.get('licensors', [])],
                            'collected_at': datetime.now().isoformat()
                        }
                        all_anime.append(anime_info)
                    
                    # Respect API rate limits
                    time.sleep(1)
                    
                elif response.status_code == 429:
                    print(f"   ⏳ Rate limited, waiting 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    print(f"   ❌ Error {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Exception on page {page}: {str(e)}")
                break
        
        print(f"✅ Collected {len(all_anime)} top anime from MyAnimeList")
        self.collected_data['mal_top'] = all_anime
        
        # Save to cache
        self._save_to_cache('mal_top_anime.json', all_anime)
        
        return all_anime
    
    def fetch_seasonal_trends(self, years=None, seasons=None):
        """
        Fetch seasonal anime data for trend analysis
        
        Benefits for ML:
        - Seasonal popularity patterns
        - Genre trends over time
        - Studio performance by season
        - Success rate prediction
        """
        if years is None:
            years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        
        if seasons is None:
            seasons = ['winter', 'spring', 'summer', 'fall']
        
        print(f"📅 Fetching seasonal data for {len(years)} years...")
        
        seasonal_data = []
        
        for year in years:
            for season in seasons:
                try:
                    print(f"   {year} {season}...")
                    
                    url = f"https://api.jikan.moe/v4/seasons/{year}/{season}"
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        anime_list = data.get('data', [])
                        
                        for anime in anime_list:
                            anime_info = {
                                'source': 'Seasonal',
                                'mal_id': anime.get('mal_id'),
                                'title': anime.get('title'),
                                'year': year,
                                'season': season,
                                'score': anime.get('score'),
                                'scored_by': anime.get('scored_by', 0),
                                'members': anime.get('members', 0),
                                'popularity': anime.get('popularity'),
                                'rank': anime.get('rank'),
                                'favorites': anime.get('favorites', 0),
                                'genres': [g['name'] for g in anime.get('genres', [])],
                                'themes': [t['name'] for t in anime.get('themes', [])],
                                'studios': [s['name'] for s in anime.get('studios', [])],
                                'producers': [p['name'] for p in anime.get('producers', [])],
                                'type': anime.get('type'),
                                'episodes': anime.get('episodes'),
                                'status': anime.get('status'),
                                'source_material': anime.get('source'),
                                'rating': anime.get('rating'),
                                'synopsis': anime.get('synopsis', ''),
                                'collected_at': datetime.now().isoformat()
                            }
                            seasonal_data.append(anime_info)
                        
                        time.sleep(1)  # Rate limiting
                        
                    elif response.status_code == 429:
                        print(f"   ⏳ Rate limited...")
                        time.sleep(5)
                        continue
                        
                except Exception as e:
                    print(f"   ❌ Error fetching {year} {season}: {str(e)}")
                    continue
        
        print(f"✅ Collected {len(seasonal_data)} seasonal anime")
        self.collected_data['seasonal'] = seasonal_data
        
        # Save to cache
        self._save_to_cache('seasonal_anime.json', seasonal_data)
        
        return seasonal_data
    
    def fetch_genre_statistics(self):
        """
        Analyze genre popularity and trends
        
        Benefits for ML:
        - Genre combination success rates
        - Trending genres
        - Genre-based scoring patterns
        """
        print("📊 Analyzing genre statistics...")
        
        # Use existing collected data
        all_anime = []
        if 'mal_top' in self.collected_data:
            all_anime.extend(self.collected_data['mal_top'])
        if 'seasonal' in self.collected_data:
            all_anime.extend(self.collected_data['seasonal'])
        
        if not all_anime:
            print("❌ No anime data available for genre analysis")
            return {}
        
        # Analyze genre patterns
        genre_stats = {}
        
        # Genre frequency
        genre_counts = {}
        genre_scores = {}
        genre_popularity = {}
        
        for anime in all_anime:
            score = anime.get('score')
            members = anime.get('members', 0)
            genres = anime.get('genres', [])
            
            for genre in genres:
                if genre not in genre_counts:
                    genre_counts[genre] = 0
                    genre_scores[genre] = []
                    genre_popularity[genre] = []
                
                genre_counts[genre] += 1
                
                if score:
                    genre_scores[genre].append(score)
                
                if members:
                    genre_popularity[genre].append(members)
        
        # Calculate statistics
        for genre in genre_counts:
            scores = genre_scores[genre]
            popularity = genre_popularity[genre]
            
            genre_stats[genre] = {
                'count': genre_counts[genre],
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'score_std': pd.Series(scores).std() if len(scores) > 1 else 0,
                'avg_members': sum(popularity) / len(popularity) if popularity else 0,
                'success_rate': len([s for s in scores if s >= 8.0]) / len(scores) if scores else 0
            }
        
        print(f"✅ Analyzed {len(genre_stats)} genres")
        
        # Save genre statistics
        self._save_to_cache('genre_statistics.json', genre_stats)
        
        return genre_stats
    
    def fetch_studio_performance(self):
        """
        Analyze studio performance metrics
        
        Benefits for ML:
        - Studio reputation scoring
        - Quality prediction based on studio
        - Studio-genre specialization
        """
        print("🏭 Analyzing studio performance...")
        
        all_anime = []
        if 'mal_top' in self.collected_data:
            all_anime.extend(self.collected_data['mal_top'])
        if 'seasonal' in self.collected_data:
            all_anime.extend(self.collected_data['seasonal'])
        
        studio_stats = {}
        
        for anime in all_anime:
            score = anime.get('score')
            members = anime.get('members', 0)
            studios = anime.get('studios', [])
            genres = anime.get('genres', [])
            
            for studio in studios:
                if studio not in studio_stats:
                    studio_stats[studio] = {
                        'anime_count': 0,
                        'scores': [],
                        'members': [],
                        'genres': []
                    }
                
                studio_stats[studio]['anime_count'] += 1
                
                if score:
                    studio_stats[studio]['scores'].append(score)
                
                if members:
                    studio_stats[studio]['members'].append(members)
                
                studio_stats[studio]['genres'].extend(genres)
        
        # Calculate studio metrics
        studio_performance = {}
        
        for studio, stats in studio_stats.items():
            if stats['anime_count'] >= 5:  # Only studios with enough data
                scores = stats['scores']
                members = stats['members']
                
                studio_performance[studio] = {
                    'anime_count': stats['anime_count'],
                    'avg_score': sum(scores) / len(scores) if scores else 0,
                    'score_consistency': 1 - (pd.Series(scores).std() / pd.Series(scores).mean()) if len(scores) > 1 else 0,
                    'avg_popularity': sum(members) / len(members) if members else 0,
                    'hit_rate': len([s for s in scores if s >= 8.5]) / len(scores) if scores else 0,
                    'top_genres': pd.Series(stats['genres']).value_counts().head(5).to_dict()
                }
        
        print(f"✅ Analyzed {len(studio_performance)} studios")
        
        # Save studio performance
        self._save_to_cache('studio_performance.json', studio_performance)
        
        return studio_performance
    
    def create_training_dataset(self):
        """
        Create a comprehensive training dataset for ML
        
        Combines all collected data into ML-ready format
        """
        print("🔧 Creating ML training dataset...")
        
        all_anime = []
        
        # Combine all data sources
        for source_key in ['mal_top', 'seasonal']:
            if source_key in self.collected_data:
                all_anime.extend(self.collected_data[source_key])
        
        if not all_anime:
            print("❌ No data available for training dataset")
            return None
        
        # Convert to DataFrame and engineer features
        df = pd.DataFrame(all_anime)
        
        # Remove duplicates based on MAL ID
        df = df.drop_duplicates(subset=['mal_id'], keep='first')
        
        # Feature engineering
        df['episode_count_log'] = df['episodes'].fillna(12).apply(lambda x: np.log1p(x if x and x > 0 else 12))
        df['members_log'] = df['members'].fillna(1000).apply(lambda x: np.log1p(x))
        df['popularity_score'] = 1 / (1 + df['popularity'].fillna(5000) / 1000)
        df['has_score'] = df['score'].notna().astype(int)
        df['genre_count'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['studio_count'] = df['studios'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['is_recent'] = (df['year'].fillna(2000) >= 2015).astype(int)
        df['synopsis_length'] = df['synopsis'].fillna('').apply(len)
        
        # One-hot encode popular genres
        popular_genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 
                         'Romance', 'Sci-Fi', 'Slice of Life', 'Supernatural', 
                         'Thriller', 'Mystery', 'Sports']
        
        for genre in popular_genres:
            df[f'has_{genre.lower().replace(" ", "_").replace("-", "_")}'] = df['genres'].apply(
                lambda x: 1 if isinstance(x, list) and genre in x else 0
            )
        
        # Studio tier based on average performance
        studio_scores = {}
        for _, row in df.iterrows():
            if row['score'] and row['studios']:
                for studio in row['studios']:
                    if studio not in studio_scores:
                        studio_scores[studio] = []
                    studio_scores[studio].append(row['score'])
        
        studio_avg_scores = {
            studio: sum(scores) / len(scores) 
            for studio, scores in studio_scores.items() 
            if len(scores) >= 3
        }
        
        def get_studio_tier(studios):
            if not studios:
                return 'Unknown'
            max_score = max([studio_avg_scores.get(studio, 7.0) for studio in studios])
            if max_score >= 8.5:
                return 'Elite'
            elif max_score >= 8.0:
                return 'Premium'
            elif max_score >= 7.5:
                return 'Good'
            else:
                return 'Standard'
        
        df['studio_tier'] = df['studios'].apply(get_studio_tier)
        
        # Save training dataset
        training_data_path = self.cache_dir / 'ml_training_dataset.csv'
        df.to_csv(training_data_path, index=False)
        
        print(f"✅ Created training dataset with {len(df)} anime and {len(df.columns)} features")
        print(f"💾 Saved to: {training_data_path}")
        
        return df
    
    def _save_to_cache(self, filename, data):
        """Save data to cache file"""
        cache_file = self.cache_dir / filename
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Cached to {cache_file}")
    
    def run_full_collection(self):
        """Run complete data collection pipeline"""
        
        print("🚀 Starting comprehensive anime data collection...")
        print("=" * 60)
        
        # Step 1: Fetch top anime
        self.fetch_mal_top_anime(max_pages=15)
        
        # Step 2: Fetch seasonal data
        self.fetch_seasonal_trends()
        
        # Step 3: Analyze genres
        self.fetch_genre_statistics()
        
        # Step 4: Analyze studios
        self.fetch_studio_performance()
        
        # Step 5: Create training dataset
        training_df = self.create_training_dataset()
        
        print("\n" + "=" * 60)
        print("✅ Data collection complete!")
        
        if training_df is not None:
            print(f"📊 Final dataset: {len(training_df)} anime")
            print(f"🎯 Features: {len(training_df.columns)} total")
            print(f"📈 Score range: {training_df['score'].min():.1f} - {training_df['score'].max():.1f}")
            print(f"📅 Year range: {int(training_df['year'].min())} - {int(training_df['year'].max())}")
            
            # Show top studios by average score
            print("\n🏆 Top studios by average score:")
            studio_scores = {}
            for _, row in training_df.iterrows():
                if row['score'] and row['studios']:
                    for studio in row['studios']:
                        if studio not in studio_scores:
                            studio_scores[studio] = []
                        studio_scores[studio].append(row['score'])
            
            top_studios = sorted(
                [(studio, sum(scores)/len(scores)) for studio, scores in studio_scores.items() if len(scores) >= 5],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            for i, (studio, avg_score) in enumerate(top_studios, 1):
                print(f"   {i:2d}. {studio}: {avg_score:.2f}")
        
        return training_df


def main():
    """Main execution function"""
    
    collector = AnimeDataCollector()
    
    print("🤖 Enhanced Anime Recommendation Data Collector")
    print("=" * 60)
    print("This script will collect anime data from external sources")
    print("to enhance your ML recommendation system.")
    print()
    
    # Run full collection
    dataset = collector.run_full_collection()
    
    print("\n🎯 Next Steps:")
    print("1. Use the collected data in your enhanced ML models")
    print("2. Train new models with the expanded dataset")
    print("3. Compare performance with local-only models")
    print("4. Enjoy much more accurate recommendations! 🎉")


if __name__ == "__main__":
    # Add numpy import for feature engineering
    import numpy as np
    main()