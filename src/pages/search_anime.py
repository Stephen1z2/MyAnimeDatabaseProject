"""
Search Anime page
"""

import streamlit as st
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.components.ui_components import AnimeSearchComponent, ErrorHandlerComponent
from src.services.database_service import database_service
from src.utils.ui_helpers import show_warning_message


def render_search_anime():
    """Render the Search Anime page"""
    st.header("🔍 Search Anime")
    
    st.markdown("""
    Search through the anime database with advanced filtering options.
    """)
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize the database first from the Database Overview page.")
        return
    
    try:
        with database_service as db:
            # Get data for form options
            genres = db.get_all_genres()
            anime_types = db.get_anime_types()
            
            # Render search form
            st.subheader("Search Parameters")
            search_params = AnimeSearchComponent.render_search_form(genres, anime_types)
            
            # Perform search on button click
            if st.button("🔍 Search", type="primary"):
                try:
                    results = db.search_anime(
                        search_term=search_params['search_term'],
                        search_type=search_params['search_type'],
                        min_score=search_params['min_score'],
                        selected_genres=search_params['selected_genres'],
                        anime_type=search_params['anime_type'],
                        limit=100
                    )
                    
                    st.subheader("Search Results")
                    AnimeSearchComponent.render_results(results)
                    
                except Exception as e:
                    ErrorHandlerComponent.handle_database_error(e, "anime search")
            
            # Show search tips
            st.subheader("💡 Search Tips")
            with st.expander("How to use the search", expanded=False):
                st.markdown("""
                **Search Types:**
                - **Contains**: Finds anime with titles containing your search term
                - **Starts With**: Finds anime with titles starting with your search term  
                - **Exact**: Finds anime with exact title match
                
                **Filters:**
                - **Minimum Score**: Only show anime with scores above this threshold
                - **Type**: Filter by anime type (TV, Movie, OVA, etc.)
                - **Genres**: Select multiple genres to narrow results
                
                **Examples:**
                - Search "attack" with "Contains" to find "Attack on Titan"
                - Use minimum score 8.0 to find highly rated anime
                - Select "Action" and "Drama" genres for specific combinations
                """)
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "initializing search page")