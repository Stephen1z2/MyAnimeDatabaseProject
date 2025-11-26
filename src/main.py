"""
Main application entry point for the refactored anime database project
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.app_config import PAGE_CONFIG, NAVIGATION_OPTIONS, HIDE_STREAMLIT_STYLE
from src.utils.ui_helpers import init_session_state
from src.pages.database_overview import render_database_overview
from src.pages.database_visualization import render_database_visualization
from src.pages.search_anime import render_search_anime
from src.pages.search_characters import render_search_characters
from src.pages.machine_learning import render_machine_learning
from src.pages.data_explorer import render_data_explorer
from src.pages.analytics import render_analytics


def main():
    """Main application function"""
    # Configure Streamlit page
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom styling
    st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # App header
    st.title("🎌 MyAnimeList Database Project")
    st.markdown("### Enhanced with advanced ML recommendations and external data integration")
    
    # Navigation sidebar
    sidebar_option = st.sidebar.selectbox("Navigation", NAVIGATION_OPTIONS)
    
    # Route to appropriate page
    route_page(sidebar_option)


def route_page(page_name: str):
    """Route to the appropriate page based on navigation selection"""
    
    if page_name == "Database Overview":
        render_database_overview()
    
    elif page_name == "Database Visualization":
        render_database_visualization()
    
    elif page_name == "Search Anime":
        render_search_anime()
    
    elif page_name == "Search Characters":
        render_search_characters()
    
    elif page_name == "💎 Hidden Gems Finder":
        render_hidden_gems_page()
    
    elif page_name == "🎯 Smart Recommendations":
        render_smart_recommendations_page()
    
    elif page_name == "🚀 Enhanced ML":
        render_enhanced_ml_page()
    
    elif page_name == "Data Explorer":
        render_data_explorer()
    
    elif page_name == "Analytics":
        render_analytics()
    
    else:
        st.error(f"Page '{page_name}' not found")


def render_data_explorer():
    """Render Data Explorer page"""
    from src.pages.data_explorer import render_data_explorer as data_explorer_page
    data_explorer_page()


def render_analytics():
    """Render Analytics page"""
    from src.pages.analytics import render_analytics as analytics_page
    analytics_page()


def render_hidden_gems_page():
    """Hidden Gems Finder - Focus on finding underrated anime"""
    st.header("💎 Hidden Gems Finder")
    st.markdown("**Discover amazing underrated anime using AI analysis!**")
    
    # Import and call the hidden gem finder from ML module
    from src.pages.machine_learning import _render_hidden_gem_finder
    _render_hidden_gem_finder()


def render_smart_recommendations_page():
    """Smart Recommendations - Personalized anime suggestions"""
    st.header("🎯 Smart Recommendations")
    st.markdown("**Get personalized anime recommendations based on your preferences!**")
    
    # Basic recommendation form
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎭 Your Preferences")
        favorite_genres = st.multiselect(
            "Favorite Genres",
            ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance", 
             "Sci-Fi", "Slice of Life", "Thriller", "Mystery", "Horror"],
            default=["Action", "Fantasy"]
        )
        
        min_score = st.slider("Minimum Score", 5.0, 9.5, 7.5, 0.1)
        min_year = st.number_input("Minimum Year", 1990, 2024, 2010)
        
    with col2:
        st.subheader("⚙️ Settings")
        max_episodes = st.number_input("Max Episodes", 1, 500, 50)
        recommendation_count = st.slider("Number of Recommendations", 5, 30, 15)
        
        recommendation_type = st.selectbox(
            "Recommendation Type",
            ["Balanced", "Hidden Gems", "Popular", "Recent Releases"]
        )
    
    if st.button("🔍 Get My Recommendations", type="primary"):
        with st.spinner("🤖 Analyzing your preferences and finding perfect matches..."):
            # Call enhanced ML recommendations
            try:
                from src.pages.enhanced_ml import EnhancedAnimeRecommendationSystem
                enhanced_system = EnhancedAnimeRecommendationSystem()
                enhanced_system.load_cached_data()
                
                user_preferences = {
                    'genres': favorite_genres,
                    'min_score': min_score,
                    'min_year': min_year,
                    'max_episodes': max_episodes,
                    'mode': recommendation_type
                }
                
                recommendations = enhanced_system.get_enhanced_recommendations(
                    user_preferences, recommendation_count
                )
                
                if recommendations:
                    st.success(f"✨ Found {len(recommendations)} perfect matches for you!")
                    
                    for i, rec in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {rec['title']} ⭐ {rec.get('score', 'N/A')}"):
                            col_a, col_b = st.columns([3, 1])
                            
                            with col_a:
                                st.write(f"**Genres:** {', '.join(rec.get('genres', [])[:5])}")
                                synopsis = rec.get('synopsis', 'No synopsis available')
                                if len(synopsis) > 300:
                                    synopsis = synopsis[:300] + "..."
                                st.write(synopsis)
                            
                            with col_b:
                                st.metric("Year", rec.get('year', 'Unknown'))
                                confidence = "🔥 High" if rec.get('recommendation_score', 0) >= 6 else "⭐ Good"
                                st.write(f"**Match:** {confidence}")
                                
                                if rec.get('mal_id'):
                                    st.markdown(f"[🔗 View on MAL](https://myanimelist.net/anime/{rec['mal_id']})")
                else:
                    st.warning("😔 No recommendations found. Try adjusting your preferences!")
                    
            except Exception as e:
                st.error("❌ Enhanced ML system not available. Using basic recommendations...")
                st.info("Run: `python external_data_collector.py` to enable enhanced recommendations")
                
                # Fallback to basic local recommendations
                from src.pages.machine_learning import _render_hidden_gem_finder
                st.markdown("---")
                st.subheader("🎯 Hidden Gems (Fallback)")
                _render_hidden_gem_finder()


def render_enhanced_ml_page():
    """Enhanced ML with External Data"""
    st.header("🚀 Enhanced Machine Learning")
    st.markdown("**Advanced ML using external data sources for superior recommendations!**")
    
    try:
        from src.pages.enhanced_ml import render_enhanced_machine_learning
        render_enhanced_machine_learning()
    except ImportError:
        st.error("❌ Enhanced ML module not available")
        st.info("Install requirements: `pip install requests scikit-learn`")
        
        # Fallback to regular ML
        st.markdown("---")
        st.subheader("📊 Standard ML Analysis")
        render_machine_learning()


def render_database_stats_page():
    """Database Statistics - Simplified overview"""
    st.header("📊 Database Statistics")
    st.markdown("**Quick overview of your anime database**")
    
    # Import key stats from database overview
    render_database_overview()


if __name__ == "__main__":
    main()