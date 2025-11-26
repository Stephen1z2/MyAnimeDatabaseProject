"""
Search Characters page
"""

import streamlit as st
import sys
import os
from sqlalchemy import func

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import get_session
from models import Character, Anime, AnimeCharacter
from src.components.ui_components import CharacterSearchComponent, MetricsComponent, ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message, show_info_message


def render_search_characters():
    """Render the Search Characters page"""
    st.header("🎭 Search Character Database")
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize and populate the database first.")
        return
    
    try:
        session = get_session()
        
        # Character search interface
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            char_search_query = st.text_input("Search by character name", placeholder="Enter character name...")
        
        with search_col2:
            char_search_type = st.selectbox("Search Type", ["Contains", "Starts With", "Exact"], key="char_search_type")
        
        # Filters
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Get available roles
            roles = session.query(AnimeCharacter.role).distinct().filter(AnimeCharacter.role.isnot(None)).all()
            role_names = ["All"] + [r[0] for r in roles if r[0]]
            selected_role = st.selectbox("Filter by Role", role_names)
        
        with filter_col2:
            # Get anime titles for filtering
            anime_titles = session.query(Anime.title).order_by(Anime.title.asc()).limit(100).all()
            anime_names = ["All"] + [a[0] for a in anime_titles]
            selected_anime = st.selectbox("Filter by Anime", anime_names)
        
        # Search button
        if st.button("🔍 Search Characters", type="primary"):
            _perform_character_search(session, char_search_query, char_search_type, selected_role, selected_anime)
        
        # Show helpful stats
        _render_character_statistics(session)
        
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "character search")
    finally:
        if 'session' in locals():
            session.close()


def _perform_character_search(session, search_query, search_type, selected_role, selected_anime):
    """Perform the character search with given parameters"""
    try:
        # Build query
        query = session.query(
            Character.name,
            Character.mal_id,
            Character.image_url,
            AnimeCharacter.role,
            Anime.title.label('anime_title'),
            Anime.score.label('anime_score')
        ).join(AnimeCharacter, Character.id == AnimeCharacter.character_id
        ).join(Anime, AnimeCharacter.anime_id == Anime.id)
        
        # Apply search filter
        if search_query:
            if search_type == "Contains":
                query = query.filter(Character.name.ilike(f"%{search_query}%"))
            elif search_type == "Starts With":
                query = query.filter(Character.name.ilike(f"{search_query}%"))
            else:
                query = query.filter(Character.name.ilike(search_query))
        
        # Apply role filter
        if selected_role != "All":
            query = query.filter(AnimeCharacter.role == selected_role)
        
        # Apply anime filter
        if selected_anime != "All":
            query = query.filter(Anime.title == selected_anime)
        
        # Order by character name
        query = query.order_by(Character.name.asc())
        
        # Execute query
        results = query.limit(50).all()
        
        st.subheader(f"Found {len(results)} characters")
        
        if results:
            _render_character_results(results)
        else:
            show_info_message("No characters found matching your criteria.")
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "performing character search")


def _render_character_results(results):
    """Render character search results"""
    # Group results by character name for better display
    char_groups = {}
    for result in results:
        char_name = result.name
        if char_name not in char_groups:
            char_groups[char_name] = []
        char_groups[char_name].append(result)
    
    for char_name, appearances in char_groups.items():
        # Get the first appearance for main character info
        main_char = appearances[0]
        
        with st.expander(f"🎭 {char_name} ({len(appearances)} anime appearances)"):
            char_col1, char_col2 = st.columns([1, 3])
            
            with char_col1:
                _render_character_image(main_char)
            
            with char_col2:
                _render_character_details(char_name, appearances, main_char)


def _render_character_image(character):
    """Render character image"""
    if character.image_url:
        try:
            st.image(character.image_url, width=150)
        except:
            st.write("📷 Image unavailable")
    else:
        st.write("📷 No image")
    
    st.write(f"**MAL ID:** {character.mal_id}")


def _render_character_details(char_name, appearances, main_char):
    """Render character details and appearances"""
    st.write(f"**Character:** {char_name}")
    st.write(f"**Appears in {len(appearances)} anime:**")
    
    # Show all anime appearances
    for appearance in appearances:
        role_emoji = "⭐" if appearance.role == "Main" else "👥" if appearance.role == "Supporting" else "🎭"
        score_text = f" (Score: {appearance.anime_score:.1f})" if appearance.anime_score else ""
        st.write(f"  {role_emoji} **{appearance.anime_title}** - {appearance.role}{score_text}")


def _render_character_statistics(session):
    """Render character database statistics"""
    st.markdown("---")
    
    try:
        total_chars = session.query(func.count(Character.id)).scalar()
        total_appearances = session.query(func.count(AnimeCharacter.anime_id)).scalar()
        
        character_stats = {
            "Total Characters": f"{total_chars:,}",
            "Total Character Appearances": f"{total_appearances:,}",
            "Avg Appearances per Character": f"{total_appearances/total_chars:.1f}" if total_chars > 0 else "0"
        }
        
        MetricsComponent.render(character_stats, columns=3)
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "calculating character statistics")