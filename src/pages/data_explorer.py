"""
Data Explorer page
"""

import streamlit as st
import pandas as pd
import sys
import os
from sqlalchemy import func

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import get_session
from models import Anime, Genre, Studio, Character, AnimeCharacter, MLFeature, anime_genres, anime_studios
from src.components.ui_components import MetricsComponent, ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message, show_info_message


def render_data_explorer():
    """Render the Data Explorer page"""
    st.header("📊 Data Explorer")
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize and populate the database first.")
        return
    
    st.markdown("""
    Browse and explore the contents of your anime database with advanced filtering and pagination.
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎬 Anime", "🎭 Genres", "🏢 Studios", "👥 Characters"])
    
    try:
        with tab1:
            _render_anime_explorer()
        
        with tab2:
            _render_genre_explorer()
        
        with tab3:
            _render_studio_explorer()
        
        with tab4:
            _render_character_explorer()
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "data exploration")


def _render_anime_explorer():
    """Render anime exploration tab"""
    st.subheader("Anime Database Explorer")
    
    session = get_session()
    try:
        # Pagination settings
        col1, col2 = st.columns(2)
        with col1:
            page_size = st.selectbox("Items per page", [25, 50, 100, 200], index=1)
        with col2:
            sort_by = st.selectbox("Sort by", ["Title", "Score", "Year", "Episodes", "Rank"])
        
        # Get total count
        total_anime = session.query(func.count(Anime.id)).scalar()
        
        # Calculate pagination
        total_pages = (total_anime - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        offset = (page - 1) * page_size
        
        # Build query
        query = session.query(Anime)
        
        # Apply sorting
        if sort_by == "Title":
            query = query.order_by(Anime.title.asc())
        elif sort_by == "Score":
            query = query.order_by(Anime.score.desc())
        elif sort_by == "Year":
            query = query.order_by(Anime.year.desc())
        elif sort_by == "Episodes":
            query = query.order_by(Anime.episodes.desc())
        else:  # Rank
            query = query.order_by(Anime.rank.asc())
        
        # Get paginated results
        anime_list = query.offset(offset).limit(page_size).all()
        
        st.info(f"Showing anime {offset + 1}-{min(offset + page_size, total_anime)} of {total_anime}")
        
        # Display results
        if anime_list:
            anime_data = []
            for anime in anime_list:
                anime_data.append({
                    "Title": anime.title,
                    "Score": f"{anime.score:.2f}" if anime.score else "N/A",
                    "Year": anime.year or "Unknown",
                    "Type": anime.type or "Unknown",
                    "Episodes": anime.episodes or "Unknown",
                    "Status": anime.status or "Unknown"
                })
            
            anime_df = pd.DataFrame(anime_data)
            st.dataframe(anime_df, use_container_width=True, hide_index=True)
        else:
            show_info_message("No anime found.")
    
    finally:
        session.close()


def _render_genre_explorer():
    """Render genre exploration tab"""
    st.subheader("Genre Analysis")
    
    session = get_session()
    try:
        # Get genre statistics
        genre_stats = session.query(
            Genre.name,
            func.count(anime_genres.c.anime_id).label('anime_count')
        ).join(anime_genres).group_by(Genre.name).order_by(
            func.count(anime_genres.c.anime_id).desc()
        ).all()
        
        if genre_stats:
            st.write(f"**Found {len(genre_stats)} genres**")
            
            genre_data = []
            for genre_name, count in genre_stats:
                genre_data.append({
                    "Genre": genre_name,
                    "Anime Count": count
                })
            
            genre_df = pd.DataFrame(genre_data)
            st.dataframe(genre_df, use_container_width=True, hide_index=True)
        else:
            show_info_message("No genre data available.")
    
    finally:
        session.close()


def _render_studio_explorer():
    """Render studio exploration tab"""
    st.subheader("Studio Analysis")
    
    session = get_session()
    try:
        # Get studio statistics
        studio_stats = session.query(
            Studio.name,
            func.count(anime_studios.c.anime_id).label('anime_count')
        ).join(anime_studios).group_by(Studio.name).order_by(
            func.count(anime_studios.c.anime_id).desc()
        ).limit(50).all()
        
        if studio_stats:
            st.write(f"**Top 50 most productive studios**")
            
            studio_data = []
            for studio_name, count in studio_stats:
                studio_data.append({
                    "Studio": studio_name,
                    "Anime Count": count
                })
            
            studio_df = pd.DataFrame(studio_data)
            st.dataframe(studio_df, use_container_width=True, hide_index=True)
        else:
            show_info_message("No studio data available.")
    
    finally:
        session.close()


def _render_character_explorer():
    """Render character exploration tab"""
    st.subheader("Character Database")
    
    session = get_session()
    try:
        # Character view options
        view_type = st.selectbox(
            "View Type",
            ["Character-Anime Relationships", "Unique Characters Only"]
        )
        
        if view_type == "Character-Anime Relationships":
            _render_character_relationships(session)
        else:
            _render_unique_characters(session)
    
    finally:
        session.close()


def _render_character_relationships(session):
    """Render character-anime relationships"""
    # Pagination
    page_size = st.selectbox("Results per page", [50, 100, 200], index=1, key="char_rel_page_size")
    
    # Get total count
    total_relationships = session.query(func.count(AnimeCharacter.character_id)).scalar()
    
    if total_relationships > 0:
        total_pages = (total_relationships - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="char_rel_page")
        offset = (page - 1) * page_size
        
        # Get character relationships
        char_rels = session.query(
            Character.name.label('character_name'),
            Character.mal_id.label('char_mal_id'),
            AnimeCharacter.role,
            Anime.title.label('anime_title'),
            Anime.score.label('anime_score')
        ).join(
            AnimeCharacter, Character.id == AnimeCharacter.character_id
        ).join(
            Anime, AnimeCharacter.anime_id == Anime.id
        ).order_by(Character.name.asc()).offset(offset).limit(page_size).all()
        
        st.info(f"Showing relationships {offset + 1}-{min(offset + page_size, total_relationships)} of {total_relationships}")
        
        if char_rels:
            char_data = []
            for char_name, char_mal_id, role, anime_title, anime_score in char_rels:
                char_data.append({
                    "Character": char_name,
                    "Role": role or "Unknown",
                    "Anime": anime_title,
                    "Anime Score": f"{anime_score:.2f}" if anime_score else "N/A",
                    "Character MAL ID": char_mal_id
                })
            
            char_df = pd.DataFrame(char_data)
            st.dataframe(char_df, use_container_width=True, hide_index=True)
    else:
        show_info_message("No character relationships found.")


def _render_unique_characters(session):
    """Render unique characters only"""
    # Get unique characters with appearance count
    char_stats = session.query(
        Character.name,
        Character.mal_id,
        func.count(AnimeCharacter.anime_id).label('appearances')
    ).join(AnimeCharacter).group_by(
        Character.id, Character.name, Character.mal_id
    ).order_by(Character.name.asc()).limit(200).all()
    
    if char_stats:
        st.info(f"Showing top 200 characters by name")
        
        char_data = []
        for char_name, char_mal_id, appearances in char_stats:
            char_data.append({
                "Character": char_name,
                "Appearances": appearances,
                "MAL ID": char_mal_id
            })
        
        char_df = pd.DataFrame(char_data)
        st.dataframe(char_df, use_container_width=True, hide_index=True)
    else:
        show_info_message("No characters found.")