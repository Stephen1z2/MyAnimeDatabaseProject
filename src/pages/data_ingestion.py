"""
Data Ingestion page
"""

import streamlit as st
import time
import sys
import os
from sqlalchemy import func

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import get_session
from models import Anime, Character, AnimeCharacter
from jikan_ingestion import run_full_ingestion, ingest_genres, ingest_top_anime, ingest_anime_characters
from src.components.ui_components import MetricsComponent, ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message, show_success_message, show_info_message


def render_data_ingestion():
    """Render the Data Ingestion page"""
    st.header("📥 Data Ingestion from Jikan API")
    
    st.markdown("""
    This section allows you to populate the database with real anime data from the Jikan API 
    (unofficial MyAnimeList API).
    """)
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize the database first from the Database Overview page.")
        return
    
    tab1, tab2 = st.tabs(["Quick Ingestion", "Custom Ingestion"])
    
    with tab1:
        _render_quick_ingestion()
    
    with tab2:
        _render_custom_ingestion()


def _render_quick_ingestion():
    """Render the Quick Ingestion tab"""
    st.subheader("Quick Database Population")
    
    show_info_message("This will fetch top anime, genres, characters, and recommendations.")
    
    num_pages = st.slider("Number of pages to fetch (25 anime per page)", 1, 5, 2)
    
    if st.button("Run Full Ingestion", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Starting ingestion...")
            progress_bar.progress(10)
            
            with st.spinner("Ingesting data from Jikan API..."):
                results = run_full_ingestion(num_pages=num_pages)
                progress_bar.progress(100)
                st.session_state.data_ingested = True
            
            status_text.empty()
            progress_bar.empty()
            
            show_success_message("Data ingestion completed!")
            
            # Display results
            results_metrics = {
                "Genres": results['genres'],
                "Anime": results['anime'], 
                "Characters": results['characters'],
                "Recommendations": results['recommendations']
            }
            MetricsComponent.render(results_metrics, columns=4)
            
            st.balloons()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            ErrorHandlerComponent.handle_api_error(e)


def _render_custom_ingestion():
    """Render the Custom Ingestion tab"""
    st.subheader("Custom Data Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Ingest Genres Only"):
            try:
                with st.spinner("Fetching genres..."):
                    count = ingest_genres()
                show_success_message(f"Ingested {count} genres!")
            except Exception as e:
                ErrorHandlerComponent.handle_api_error(e)
    
    with col2:
        page_num = st.number_input("Page number", min_value=1, max_value=100, value=1)
        if st.button("Ingest Top Anime Page"):
            try:
                with st.spinner(f"Fetching page {page_num}..."):
                    count = ingest_top_anime(page=page_num)
                show_success_message(f"Ingested {count} anime from page {page_num}!")
            except Exception as e:
                ErrorHandlerComponent.handle_api_error(e)
    
    # Character ingestion section
    _render_character_ingestion()


def _render_character_ingestion():
    """Render the Character Ingestion section"""
    st.subheader("Character Data Ingestion")
    
    char_col1, char_col2 = st.columns(2)
    
    try:
        # Check how many anime need characters
        session = get_session()
        anime_with_chars = session.query(func.count(func.distinct(AnimeCharacter.anime_id))).scalar()
        total_anime = session.query(func.count(Anime.id)).scalar()
        anime_without_chars = total_anime - anime_with_chars
        session.close()
        
        with char_col1:
            character_stats = f"""📊 **Character Status:**
- {anime_with_chars:,} anime have characters
- {anime_without_chars:,} anime need characters"""
            show_info_message(character_stats)
            
            if anime_without_chars > 0:
                max_batch = min(100, anime_without_chars)
                num_anime_chars = st.number_input(
                    "Number of anime to process", 
                    min_value=1, 
                    max_value=max_batch, 
                    value=min(25, max_batch), 
                    key="char_anime_count"
                )
                show_info_message(f"Will ingest characters for {num_anime_chars} anime **without character data**")
            else:
                show_success_message("🎉 All anime already have character data!")
        
        with char_col2:
            if anime_without_chars > 0 and st.button("Ingest Characters", type="secondary"):
                _process_character_ingestion(anime_without_chars, num_anime_chars)
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "checking character status")


def _process_character_ingestion(anime_without_chars, num_anime_chars):
    """Process character ingestion for selected anime"""
    try:
        session = get_session()
        # Get anime that DON'T have character data yet
        anime_list = session.query(Anime).outerjoin(
            AnimeCharacter, Anime.id == AnimeCharacter.anime_id
        ).filter(AnimeCharacter.anime_id.is_(None)
        ).order_by(Anime.rank.asc()).limit(num_anime_chars).all()
        session.close()
        
        if not anime_list:
            st.error("No anime found without character data!")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_characters = 0
        
        for i, anime in enumerate(anime_list):
            status_text.text(f"Processing {anime.title} ({i+1}/{len(anime_list)})")
            progress = (i + 1) / len(anime_list)
            progress_bar.progress(progress)
            
            try:
                count = ingest_anime_characters(anime.mal_id)
                total_characters += count
                
                # Small delay for API rate limiting
                time.sleep(1.2)
                
            except Exception as e:
                st.error(f"Error processing {anime.title}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        show_success_message(f"✅ Character ingestion completed! Added {total_characters:,} characters for {len(anime_list)} anime.")
        
        # Show updated statistics
        session = get_session()
        new_anime_with_chars = session.query(func.count(func.distinct(AnimeCharacter.anime_id))).scalar()
        new_anime_without_chars = session.query(func.count(Anime.id)).scalar() - new_anime_with_chars
        session.close()
        
        updated_metrics = {
            "Anime with Characters": new_anime_with_chars,
            "Anime without Characters": new_anime_without_chars,
            "Characters Added": total_characters
        }
        MetricsComponent.render(updated_metrics, columns=3)
        
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "character ingestion")