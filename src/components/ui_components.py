"""
Reusable UI components for the anime database application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Optional

from ..utils.ui_helpers import (
    create_metrics_row, create_bar_chart, create_pie_chart, 
    create_histogram, show_error_message, format_number
)


class DatabaseStatsComponent:
    """Component for displaying database statistics"""
    
    @staticmethod
    def render(stats: Dict[str, int]) -> None:
        """Render database statistics table and chart"""
        if not stats:
            st.info("No database statistics available")
            return
        
        # Create DataFrame
        stats_df = pd.DataFrame([
            {"Table": k, "Records": v, "Type": "Junction" if "_" in k else "Entity"} 
            for k, v in stats.items() if v > 0
        ])
        stats_df = stats_df.sort_values("Records", ascending=False)
        
        # Display table
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Display chart
        fig = create_bar_chart(
            stats_df.sort_values("Records", ascending=True), 
            x="Records", 
            y="Table",
            title="Records per Table",
            orientation='h',
            color_col="Type"
        )
        st.plotly_chart(fig, use_container_width=True)


class MetricsComponent:
    """Component for displaying metrics in columns"""
    
    @staticmethod
    def render(metrics: Dict[str, Any], columns: int = 4) -> None:
        """Render metrics in columns"""
        create_metrics_row(metrics, columns)


class AnimeSearchComponent:
    """Component for anime search functionality"""
    
    @staticmethod
    def render_search_form(genres: List[str], anime_types: List[str]) -> Dict[str, Any]:
        """Render search form and return search parameters"""
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("🔍 Search anime title:")
            search_type = st.selectbox(
                "Search type:",
                ["contains", "starts_with", "exact"],
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        with col2:
            min_score = st.slider("Minimum score:", 0.0, 10.0, 0.0, 0.1)
            anime_type = st.selectbox("Type:", ["All"] + anime_types)
        
        # Genre filter
        selected_genres = st.multiselect("Filter by genres:", genres)
        
        return {
            'search_term': search_term,
            'search_type': search_type,
            'min_score': min_score,
            'anime_type': anime_type,
            'selected_genres': selected_genres
        }
    
    @staticmethod
    def render_results(results: List[Any]) -> None:
        """Render search results"""
        if not results:
            st.info("No anime found matching your criteria.")
            return
        
        st.success(f"Found {len(results)} anime")
        
        # Create results DataFrame
        results_data = []
        for anime in results:
            results_data.append({
                "Title": anime.title,
                "Type": anime.type or "Unknown",
                "Episodes": anime.episodes or "Unknown",
                "Score": f"{anime.score:.2f}" if anime.score else "N/A",
                "Year": anime.year or "Unknown",
                "MAL ID": anime.mal_id
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)


class CharacterSearchComponent:
    """Component for character search functionality"""
    
    @staticmethod
    def render_search_form() -> Dict[str, Any]:
        """Render character search form"""
        col1, col2 = st.columns(2)
        
        with col1:
            search_term = st.text_input("🔍 Character name:")
            role_filter = st.selectbox("Role:", ["All", "Main", "Supporting"])
        
        with col2:
            anime_filter = st.text_input("🎬 Anime title (optional):")
            limit = st.selectbox("Results limit:", [50, 100, 200, 500], index=1)
        
        return {
            'search_term': search_term,
            'role_filter': role_filter,
            'anime_filter': anime_filter,
            'limit': limit
        }
    
    @staticmethod
    def render_results(results: List[Any]) -> None:
        """Render character search results"""
        if not results:
            st.info("No characters found matching your criteria.")
            return
        
        st.success(f"Found {len(results)} character-anime relationships")
        
        # Create results DataFrame
        results_data = []
        for char_name, image_url, role, anime_title, anime_score, char_mal_id, anime_mal_id in results:
            results_data.append({
                "Character": char_name,
                "Role": role or "Unknown",
                "Anime": anime_title,
                "Anime Score": f"{anime_score:.2f}" if anime_score else "N/A",
                "Character MAL ID": char_mal_id,
                "Anime MAL ID": anime_mal_id
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True, hide_index=True)


class DistributionChartComponent:
    """Component for data distribution charts"""
    
    @staticmethod
    def render_score_distribution(data: pd.DataFrame) -> None:
        """Render score distribution histogram"""
        if 'Score' in data.columns and not data['Score'].isna().all():
            fig = create_histogram(data, 'Score', "Anime Score Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No score data available for distribution analysis")
    
    @staticmethod
    def render_type_distribution(data: pd.DataFrame) -> None:
        """Render anime type distribution pie chart"""
        if 'Type' in data.columns:
            type_counts = data['Type'].value_counts()
            type_df = pd.DataFrame({
                'Type': type_counts.index,
                'Count': type_counts.values
            })
            fig = create_pie_chart(type_df, 'Count', 'Type', "Anime Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No type data available for distribution analysis")


class TopEntitiesComponent:
    """Component for displaying top genres, studios, etc."""
    
    @staticmethod
    def render_top_genres(genre_data: List[tuple]) -> None:
        """Render top genres chart"""
        if not genre_data:
            st.info("No genre data available.")
            return
        
        genre_df = pd.DataFrame(genre_data, columns=['Genre', 'Anime Count'])
        fig = create_bar_chart(
            genre_df, 
            x="Anime Count", 
            y="Genre",
            title="Most Popular Genres",
            orientation='h',
            color_col="Anime Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_top_studios(studio_data: List[tuple]) -> None:
        """Render top studios chart"""
        if not studio_data:
            st.info("No studio data available.")
            return
        
        studio_df = pd.DataFrame(studio_data, columns=['Studio', 'Anime Count'])
        fig = create_bar_chart(
            studio_df, 
            x="Anime Count", 
            y="Studio",
            title="Most Productive Studios",
            orientation='h',
            color_col="Anime Count"
        )
        st.plotly_chart(fig, use_container_width=True)


class ErrorHandlerComponent:
    """Component for error handling and display"""
    
    @staticmethod
    def handle_database_error(error: Exception, operation: str = "database operation") -> None:
        """Handle and display database errors"""
        show_error_message(error, f"during {operation}")
        st.write("**Troubleshooting tips:**")
        st.write("1. Ensure the database is initialized")
        st.write("2. Check if data has been ingested")
        st.write("3. Try refreshing the page")
    
    @staticmethod
    def handle_api_error(error: Exception) -> None:
        """Handle and display API errors"""
        show_error_message(error, "during API operation")
        st.write("**Possible causes:**")
        st.write("1. Network connectivity issues")
        st.write("2. API rate limiting")
        st.write("3. Invalid API response")


class LoadingComponent:
    """Component for loading states and progress"""
    
    @staticmethod
    def with_spinner(message: str, func, *args, **kwargs):
        """Execute function with loading spinner"""
        with st.spinner(message):
            return func(*args, **kwargs)