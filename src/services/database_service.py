"""
Data access layer - handles all database queries and operations
"""

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Tuple, Any

# Import models - adjust path as needed
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Anime, Genre, Studio, Theme, Character, Review, 
    Recommendation, MLFeature, AnimeCharacter, 
    anime_genres, anime_studios, anime_themes
)
from database import get_session, get_table_counts


class DatabaseService:
    """Service class for database operations"""
    
    def __init__(self):
        self.session: Optional[Session] = None
    
    def __enter__(self):
        self.session = get_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    def get_table_statistics(self) -> Dict[str, int]:
        """Get row counts for all tables"""
        return get_table_counts()
    
    def get_anime_data(self, limit: int = 1000) -> List[Tuple]:
        """Get anime data for analysis"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        return self.session.query(
            Anime.score, Anime.episodes, Anime.year, 
            Anime.type, Anime.title
        ).filter(
            Anime.score.isnot(None)
        ).limit(limit).all()
    
    def get_top_genres(self, limit: int = 10) -> List[Tuple]:
        """Get most popular genres"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        return self.session.query(
            Genre.name, 
            func.count(anime_genres.c.anime_id).label('anime_count')
        ).join(anime_genres).group_by(Genre.name).order_by(
            func.count(anime_genres.c.anime_id).desc()
        ).limit(limit).all()
    
    def get_top_studios(self, limit: int = 10) -> List[Tuple]:
        """Get most productive studios"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        return self.session.query(
            Studio.name, 
            func.count(anime_studios.c.anime_id).label('anime_count')
        ).join(anime_studios).group_by(Studio.name).order_by(
            func.count(anime_studios.c.anime_id).desc()
        ).limit(limit).all()
    
    def get_character_appearances(self, limit: int = 15) -> List[Tuple]:
        """Get characters with multiple appearances"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        return self.session.query(
            func.count(AnimeCharacter.anime_id).label('appearances'),
            Character.name
        ).join(Character).group_by(Character.name).order_by(
            func.count(AnimeCharacter.anime_id).desc()
        ).limit(limit).all()
    
    def search_anime(
        self, 
        search_term: str = "", 
        search_type: str = "contains",
        min_score: float = 0.0,
        selected_genres: List[str] = None,
        anime_type: str = "All",
        limit: int = 100
    ) -> List[Anime]:
        """Search anime with filters"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        query = self.session.query(Anime)
        
        # Apply search term filter
        if search_term:
            if search_type == "contains":
                query = query.filter(Anime.title.ilike(f"%{search_term}%"))
            elif search_type == "starts_with":
                query = query.filter(Anime.title.ilike(f"{search_term}%"))
            elif search_type == "exact":
                query = query.filter(Anime.title.ilike(search_term))
        
        # Apply score filter
        if min_score > 0:
            query = query.filter(Anime.score >= min_score)
        
        # Apply type filter
        if anime_type != "All":
            query = query.filter(Anime.type == anime_type)
        
        # Apply genre filter
        if selected_genres:
            query = query.join(Anime.genres).filter(Genre.name.in_(selected_genres))
        
        return query.limit(limit).all()
    
    def search_characters(
        self,
        search_term: str = "",
        role_filter: str = "All",
        anime_filter: str = "",
        limit: int = 100
    ) -> List[Tuple]:
        """Search characters with filters"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        query = self.session.query(
            Character.name,
            Character.image_url,
            AnimeCharacter.role,
            Anime.title,
            Anime.score,
            Character.mal_id,
            Anime.mal_id.label('anime_mal_id')
        ).join(
            AnimeCharacter, Character.id == AnimeCharacter.character_id
        ).join(
            Anime, AnimeCharacter.anime_id == Anime.id
        )
        
        # Apply filters
        if search_term:
            query = query.filter(Character.name.ilike(f"%{search_term}%"))
        
        if role_filter != "All":
            query = query.filter(AnimeCharacter.role == role_filter)
        
        if anime_filter:
            query = query.filter(Anime.title.ilike(f"%{anime_filter}%"))
        
        return query.order_by(Character.name).limit(limit).all()
    
    def get_recommendation_stats(self) -> Optional[Tuple]:
        """Get recommendation network statistics"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        return self.session.query(
            func.avg(Recommendation.votes).label('avg_votes'),
            func.max(Recommendation.votes).label('max_votes'),
            func.count(Recommendation.id).label('total_recs')
        ).first()
    
    def get_anime_types(self) -> List[str]:
        """Get all unique anime types"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        types = self.session.query(Anime.type).distinct().all()
        return [t[0] for t in types if t[0]]
    
    def get_all_genres(self) -> List[str]:
        """Get all genre names"""
        if not self.session:
            raise RuntimeError("Database session not initialized")
        
        genres = self.session.query(Genre.name).all()
        return [g[0] for g in genres]
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database metrics"""
        counts = self.get_table_statistics()
        
        metrics = {
            'total_records': sum(counts.values()),
            'main_entities': counts.get('anime', 0) + counts.get('characters', 0) + 
                           counts.get('genres', 0) + counts.get('studios', 0),
            'relationships': counts.get('anime_genres', 0) + counts.get('anime_studios', 0) + 
                          counts.get('anime_characters', 0)
        }
        
        # Calculate density metrics
        anime_count = counts.get('anime', 0)
        if anime_count > 0:
            metrics.update({
                'genres_per_anime': counts.get('anime_genres', 0) / anime_count,
                'studios_per_anime': counts.get('anime_studios', 0) / anime_count,
                'chars_per_anime': counts.get('anime_characters', 0) / anime_count,
                'reviews_per_anime': counts.get('reviews', 0) / anime_count
            })
        
        return metrics


# Global service instance for easy access
database_service = DatabaseService()