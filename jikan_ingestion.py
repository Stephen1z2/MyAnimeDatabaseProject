import requests
import time
from datetime import datetime
from sqlalchemy.exc import IntegrityError
from models import Anime, Genre, Studio, Theme, Character, Review, Recommendation, AnimeCharacter
from database import get_session

JIKAN_API_BASE = "https://api.jikan.moe/v4"
RATE_LIMIT_DELAY = 1

def make_jikan_request(endpoint):
    url = f"{JIKAN_API_BASE}/{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        time.sleep(RATE_LIMIT_DELAY)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint}: {e}")
        return None

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
    except:
        return None

def ingest_genres():
    session = get_session()
    try:
        data = make_jikan_request("genres/anime")
        if not data or 'data' not in data:
            return 0
        
        count = 0
        for genre_data in data['data']:
            try:
                genre = Genre(
                    mal_id=genre_data['mal_id'],
                    name=genre_data['name']
                )
                session.merge(genre)
                count += 1
            except Exception as e:
                print(f"Error adding genre {genre_data.get('name')}: {e}")
        
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        print(f"Error ingesting genres: {e}")
        return 0
    finally:
        session.close()

def ingest_top_anime(page=1, limit=25):
    session = get_session()
    try:
        data = make_jikan_request(f"top/anime?page={page}&limit={limit}")
        if not data or 'data' not in data:
            return 0
        
        count = 0
        for anime_data in data['data']:
            try:
                aired = anime_data.get('aired', {})
                
                anime = Anime(
                    mal_id=anime_data['mal_id'],
                    title=anime_data.get('title', 'Unknown')[:500],
                    title_english=anime_data.get('title_english', '')[:500] if anime_data.get('title_english') else None,
                    title_japanese=anime_data.get('title_japanese', '')[:500] if anime_data.get('title_japanese') else None,
                    type=anime_data.get('type'),
                    source=anime_data.get('source'),
                    episodes=anime_data.get('episodes'),
                    status=anime_data.get('status'),
                    airing=anime_data.get('airing', False),
                    aired_from=parse_date(aired.get('from')) if aired else None,
                    aired_to=parse_date(aired.get('to')) if aired else None,
                    duration=anime_data.get('duration'),
                    rating=anime_data.get('rating'),
                    score=anime_data.get('score'),
                    scored_by=anime_data.get('scored_by'),
                    rank=anime_data.get('rank'),
                    popularity=anime_data.get('popularity'),
                    members=anime_data.get('members'),
                    favorites=anime_data.get('favorites'),
                    synopsis=anime_data.get('synopsis'),
                    background=anime_data.get('background'),
                    season=anime_data.get('season'),
                    year=anime_data.get('year'),
                    image_url=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                    trailer_url=anime_data.get('trailer', {}).get('url')
                )
                
                existing_anime = session.query(Anime).filter_by(mal_id=anime_data['mal_id']).first()
                if existing_anime:
                    for key, value in anime.__dict__.items():
                        if key != '_sa_instance_state' and key != 'id':
                            setattr(existing_anime, key, value)
                    anime = existing_anime
                else:
                    session.add(anime)
                
                session.flush()
                
                for genre_data in anime_data.get('genres', []):
                    genre = session.query(Genre).filter_by(mal_id=genre_data['mal_id']).first()
                    if genre and genre not in anime.genres:
                        anime.genres.append(genre)
                
                for studio_data in anime_data.get('studios', []):
                    studio = session.query(Studio).filter_by(mal_id=studio_data['mal_id']).first()
                    if not studio:
                        studio = Studio(
                            mal_id=studio_data['mal_id'],
                            name=studio_data['name'][:200]
                        )
                        session.add(studio)
                        session.flush()
                    if studio not in anime.studios:
                        anime.studios.append(studio)
                
                for theme_data in anime_data.get('themes', []):
                    theme = session.query(Theme).filter_by(mal_id=theme_data['mal_id']).first()
                    if not theme:
                        theme = Theme(
                            mal_id=theme_data['mal_id'],
                            name=theme_data['name'][:100]
                        )
                        session.add(theme)
                        session.flush()
                    if theme not in anime.themes:
                        anime.themes.append(theme)
                
                session.commit()
                count += 1
                
            except Exception as e:
                print(f"Error adding anime {anime_data.get('title')}: {e}")
                session.rollback()
                continue
        
        return count
    except Exception as e:
        session.rollback()
        print(f"Error ingesting anime: {e}")
        return 0
    finally:
        session.close()

def ingest_anime_characters(anime_mal_id):
    session = get_session()
    try:
        anime = session.query(Anime).filter_by(mal_id=anime_mal_id).first()
        if not anime:
            return 0
        
        data = make_jikan_request(f"anime/{anime_mal_id}/characters")
        if not data or 'data' not in data:
            return 0
        
        count = 0
        for char_data in data['data'][:10]:
            try:
                character_info = char_data.get('character', {})
                role = char_data.get('role', 'Unknown')
                
                character = session.query(Character).filter_by(mal_id=character_info['mal_id']).first()
                if not character:
                    character = Character(
                        mal_id=character_info['mal_id'],
                        name=character_info.get('name', 'Unknown')[:300],
                        image_url=character_info.get('images', {}).get('jpg', {}).get('image_url')
                    )
                    session.add(character)
                    session.flush()
                
                existing_assoc = session.query(AnimeCharacter).filter_by(
                    anime_id=anime.id,
                    character_id=character.id
                ).first()
                
                if not existing_assoc:
                    anime_char_assoc = AnimeCharacter(
                        anime_id=anime.id,
                        character_id=character.id,
                        role=role
                    )
                    session.add(anime_char_assoc)
                    count += 1
                    
            except Exception as e:
                print(f"Error adding character: {e}")
                continue
        
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        print(f"Error ingesting characters: {e}")
        return 0
    finally:
        session.close()

def ingest_anime_recommendations(anime_mal_id):
    session = get_session()
    try:
        anime = session.query(Anime).filter_by(mal_id=anime_mal_id).first()
        if not anime:
            return 0
        
        data = make_jikan_request(f"anime/{anime_mal_id}/recommendations")
        if not data or 'data' not in data:
            return 0
        
        count = 0
        for rec_data in data['data'][:5]:
            try:
                rec_anime_data = rec_data.get('entry', {})
                rec_anime = session.query(Anime).filter_by(mal_id=rec_anime_data['mal_id']).first()
                
                if rec_anime:
                    recommendation = Recommendation(
                        anime_id=anime.id,
                        recommended_anime_id=rec_anime.id,
                        votes=rec_data.get('votes', 0)
                    )
                    
                    existing = session.query(Recommendation).filter_by(
                        anime_id=anime.id,
                        recommended_anime_id=rec_anime.id
                    ).first()
                    
                    if not existing:
                        session.add(recommendation)
                        count += 1
            except Exception as e:
                print(f"Error adding recommendation: {e}")
        
        session.commit()
        return count
    except Exception as e:
        session.rollback()
        print(f"Error ingesting recommendations: {e}")
        return 0
    finally:
        session.close()

def ingest_seasonal_anime(year, season):
    session = get_session()
    try:
        data = make_jikan_request(f"seasons/{year}/{season}")
        if not data or 'data' not in data:
            return 0
        
        count = 0
        for anime_data in data['data'][:20]:
            try:
                aired = anime_data.get('aired', {})
                
                anime = Anime(
                    mal_id=anime_data['mal_id'],
                    title=anime_data.get('title', 'Unknown')[:500],
                    title_english=anime_data.get('title_english', '')[:500] if anime_data.get('title_english') else None,
                    title_japanese=anime_data.get('title_japanese', '')[:500] if anime_data.get('title_japanese') else None,
                    type=anime_data.get('type'),
                    source=anime_data.get('source'),
                    episodes=anime_data.get('episodes'),
                    status=anime_data.get('status'),
                    airing=anime_data.get('airing', False),
                    aired_from=parse_date(aired.get('from')) if aired else None,
                    aired_to=parse_date(aired.get('to')) if aired else None,
                    duration=anime_data.get('duration'),
                    rating=anime_data.get('rating'),
                    score=anime_data.get('score'),
                    scored_by=anime_data.get('scored_by'),
                    rank=anime_data.get('rank'),
                    popularity=anime_data.get('popularity'),
                    members=anime_data.get('members'),
                    favorites=anime_data.get('favorites'),
                    synopsis=anime_data.get('synopsis'),
                    background=anime_data.get('background'),
                    season=anime_data.get('season'),
                    year=anime_data.get('year'),
                    image_url=anime_data.get('images', {}).get('jpg', {}).get('large_image_url'),
                    trailer_url=anime_data.get('trailer', {}).get('url')
                )
                
                existing_anime = session.query(Anime).filter_by(mal_id=anime_data['mal_id']).first()
                if existing_anime:
                    for key, value in anime.__dict__.items():
                        if key != '_sa_instance_state' and key != 'id':
                            setattr(existing_anime, key, value)
                    anime = existing_anime
                else:
                    session.add(anime)
                
                session.flush()
                
                for genre_data in anime_data.get('genres', []):
                    genre = session.query(Genre).filter_by(mal_id=genre_data['mal_id']).first()
                    if genre and genre not in anime.genres:
                        anime.genres.append(genre)
                
                for studio_data in anime_data.get('studios', []):
                    studio = session.query(Studio).filter_by(mal_id=studio_data['mal_id']).first()
                    if not studio:
                        studio = Studio(
                            mal_id=studio_data['mal_id'],
                            name=studio_data['name'][:200]
                        )
                        session.add(studio)
                        session.flush()
                    if studio not in anime.studios:
                        anime.studios.append(studio)
                
                session.commit()
                count += 1
                
            except Exception as e:
                print(f"Error adding seasonal anime: {e}")
                session.rollback()
                continue
        
        return count
    except Exception as e:
        session.rollback()
        print(f"Error ingesting seasonal anime: {e}")
        return 0
    finally:
        session.close()

def run_full_ingestion(num_pages=2):
    results = {
        'genres': 0,
        'anime': 0,
        'characters': 0,
        'recommendations': 0
    }
    
    print("Ingesting genres...")
    results['genres'] = ingest_genres()
    print(f"Ingested {results['genres']} genres")
    
    print(f"Ingesting top anime ({num_pages} pages)...")
    for page in range(1, num_pages + 1):
        count = ingest_top_anime(page=page, limit=25)
        results['anime'] += count
        print(f"Page {page}: Ingested {count} anime")
    
    session = get_session()
    anime_list = session.query(Anime).limit(20).all()
    session.close()
    
    print("Ingesting characters for anime...")
    for anime in anime_list[:10]:
        count = ingest_anime_characters(anime.mal_id)
        results['characters'] += count
    
    print("Ingesting recommendations...")
    for anime in anime_list[:10]:
        count = ingest_anime_recommendations(anime.mal_id)
        results['recommendations'] += count
    
    return results
