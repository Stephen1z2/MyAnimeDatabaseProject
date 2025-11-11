from sqlalchemy import Column, Integer, String, Float, Text, Date, ForeignKey, Table, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

anime_genres = Table(
    'anime_genres',
    Base.metadata,
    Column('anime_id', Integer, ForeignKey('anime.id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.id'), primary_key=True)
)

anime_studios = Table(
    'anime_studios',
    Base.metadata,
    Column('anime_id', Integer, ForeignKey('anime.id'), primary_key=True),
    Column('studio_id', Integer, ForeignKey('studios.id'), primary_key=True)
)

anime_themes = Table(
    'anime_themes',
    Base.metadata,
    Column('anime_id', Integer, ForeignKey('anime.id'), primary_key=True),
    Column('theme_id', Integer, ForeignKey('themes.id'), primary_key=True)
)

class AnimeCharacter(Base):
    __tablename__ = 'anime_characters'
    
    anime_id = Column(Integer, ForeignKey('anime.id'), primary_key=True)
    character_id = Column(Integer, ForeignKey('characters.id'), primary_key=True)
    role = Column(String(50))
    
    anime = relationship('Anime', back_populates='anime_character_associations')
    character = relationship('Character', back_populates='anime_character_associations')

class Anime(Base):
    __tablename__ = 'anime'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False, index=True)
    title_english = Column(String(500))
    title_japanese = Column(String(500))
    type = Column(String(50))
    source = Column(String(100))
    episodes = Column(Integer)
    status = Column(String(50))
    airing = Column(Boolean)
    aired_from = Column(Date)
    aired_to = Column(Date)
    duration = Column(String(100))
    rating = Column(String(50))
    score = Column(Float, index=True)
    scored_by = Column(Integer)
    rank = Column(Integer, index=True)
    popularity = Column(Integer, index=True)
    members = Column(Integer)
    favorites = Column(Integer)
    synopsis = Column(Text)
    background = Column(Text)
    season = Column(String(20))
    year = Column(Integer, index=True)
    image_url = Column(String(500))
    trailer_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    genres = relationship('Genre', secondary=anime_genres, back_populates='animes')
    studios = relationship('Studio', secondary=anime_studios, back_populates='animes')
    themes = relationship('Theme', secondary=anime_themes, back_populates='animes')
    anime_character_associations = relationship('AnimeCharacter', back_populates='anime')
    reviews = relationship('Review', back_populates='anime')
    recommendations_from = relationship('Recommendation', foreign_keys='Recommendation.anime_id', back_populates='anime')
    ml_features = relationship('MLFeature', back_populates='anime', uselist=False)
    
    @property
    def characters(self):
        return [assoc.character for assoc in self.anime_character_associations]

class Genre(Base):
    __tablename__ = 'genres'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False, unique=True, index=True)
    
    animes = relationship('Anime', secondary=anime_genres, back_populates='genres')

class Studio(Base):
    __tablename__ = 'studios'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(200), nullable=False, unique=True, index=True)
    
    animes = relationship('Anime', secondary=anime_studios, back_populates='studios')

class Theme(Base):
    __tablename__ = 'themes'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False)
    name = Column(String(100), nullable=False, unique=True, index=True)
    
    animes = relationship('Anime', secondary=anime_themes, back_populates='themes')

class Character(Base):
    __tablename__ = 'characters'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False, index=True)
    name = Column(String(300), nullable=False, index=True)
    image_url = Column(String(500))
    
    anime_character_associations = relationship('AnimeCharacter', back_populates='character')
    
    @property
    def animes(self):
        return [assoc.anime for assoc in self.anime_character_associations]

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True)
    mal_id = Column(Integer, unique=True, nullable=False)
    anime_id = Column(Integer, ForeignKey('anime.id'), nullable=False, index=True)
    username = Column(String(100))
    date = Column(DateTime)
    score = Column(Integer)
    review_text = Column(Text)
    helpful_count = Column(Integer)
    sentiment_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    anime = relationship('Anime', back_populates='reviews')

class Recommendation(Base):
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    anime_id = Column(Integer, ForeignKey('anime.id'), nullable=False, index=True)
    recommended_anime_id = Column(Integer, ForeignKey('anime.id'), nullable=False, index=True)
    votes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    anime = relationship('Anime', foreign_keys=[anime_id], back_populates='recommendations_from')
    recommended_anime = relationship('Anime', foreign_keys=[recommended_anime_id])

class MLFeature(Base):
    __tablename__ = 'ml_features'
    
    id = Column(Integer, primary_key=True)
    anime_id = Column(Integer, ForeignKey('anime.id'), nullable=False, unique=True, index=True)
    synopsis_category = Column(String(100))
    synopsis_embedding = Column(Text)
    predicted_rating = Column(Float)
    genre_cluster = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    anime = relationship('Anime', back_populates='ml_features')
