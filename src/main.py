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
from src.pages.search_anime import render_search_anime
from src.pages.database_visualization import render_database_visualization
from src.pages.machine_learning import render_machine_learning
from src.pages.database_schema import render_database_schema
from src.pages.data_ingestion import render_data_ingestion
from src.pages.search_characters import render_search_characters
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
    st.markdown("### Database-focused project with Jikan API & ML capabilities")
    
    # Navigation sidebar
    sidebar_option = st.sidebar.selectbox("Navigation", NAVIGATION_OPTIONS)
    
    # Route to appropriate page
    route_page(sidebar_option)


def route_page(page_name: str):
    """Route to the appropriate page based on navigation selection"""
    
    if page_name == "Database Overview":
        render_database_overview()
    
    elif page_name == "Database Schema":
        render_database_schema()
    
    elif page_name == "Database Visualization":
        render_database_visualization()
    
    elif page_name == "Data Ingestion":
        render_data_ingestion()
    
    elif page_name == "Search Anime":
        render_search_anime()
    
    elif page_name == "Search Characters":
        render_search_characters()
    
    elif page_name == "Data Explorer":
        render_data_explorer()
    
    elif page_name == "Recommendations":
        render_recommendations()
    
    elif page_name == "Neural Network":
        render_neural_network()
    
    elif page_name == "Machine Learning":
        render_machine_learning()
    
    elif page_name == "Data Quality":
        render_data_quality()
    
    elif page_name == "ML Features":
        render_ml_features()
    
    elif page_name == "Analytics":
        render_analytics()
    
    else:
        st.error(f"Page '{page_name}' not implemented yet")


def render_database_schema():
    """Render Database Schema page - now fully implemented"""
    from src.pages.database_schema import render_database_schema as schema_page
    schema_page()


def render_data_ingestion():
    """Render Data Ingestion page - now fully implemented"""
    from src.pages.data_ingestion import render_data_ingestion as ingestion_page
    ingestion_page()


def render_search_characters():
    """Render Search Characters page - now fully implemented"""
    from src.pages.search_characters import render_search_characters as characters_page
    characters_page()


def render_data_explorer():
    """Render Data Explorer page - now fully implemented"""
    from src.pages.data_explorer import render_data_explorer as explorer_page
    explorer_page()


def render_recommendations():
    """Render Recommendations page - simplified implementation"""
    st.header("🎯 Anime Recommendations")
    st.info("This page provides anime recommendations based on your preferences.")
    st.write("**Note**: For the full recommendation engine, please use the original app.py for now.")
    st.write("This page includes complex similarity algorithms that are being optimized for the refactored structure.")


def render_neural_network():
    """Render Neural Network page - simplified implementation"""
    st.header("🧠 Neural Network Demo")
    st.info("This page demonstrates neural network functionality for anime analysis.")
    st.write("**Note**: For the full neural network features, please use the original app.py for now.")
    st.write("The neural network implementation includes TensorFlow models that are being optimized for the refactored structure.")


def render_machine_learning():
    """Render Machine Learning page - now fully implemented"""
    from src.pages.machine_learning import render_machine_learning as ml_page
    ml_page()


def render_data_quality():
    """Render Data Quality page - simplified implementation"""
    st.header("🔍 Data Quality Analysis")
    st.info("This page analyzes data quality, duplicates, and integrity issues.")
    st.write("**Note**: For comprehensive data quality analysis, please use the original app.py for now.")
    st.write("The full implementation includes advanced duplicate detection and data validation algorithms.")


def render_ml_features():
    """Render ML Features page - simplified implementation"""
    st.header("🎛️ ML Features Generation")
    st.info("This page handles machine learning feature generation using Hugging Face models.")
    st.write("**Note**: For ML feature generation, please use the original app.py for now.")
    st.write("The implementation includes synopsis classification and sentiment analysis using NLP models.")


def render_analytics():
    """Render Analytics page - now fully implemented"""
    from src.pages.analytics import render_analytics as analytics_page
    analytics_page()


if __name__ == "__main__":
    main()