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
        """Render search results with detailed information and images"""
        if not results:
            st.info("No anime found matching your criteria.")
            return
        
        st.success(f"Found {len(results)} anime")
        
        # Display results with expandable details
        for anime in results:
            # Create a container for each anime
            with st.container():
                # Main info in columns
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### {anime.title}")
                    if anime.title_english and anime.title_english != anime.title:
                        st.markdown(f"*English: {anime.title_english}*")
                
                with col2:
                    if anime.score:
                        st.metric("Score", f"{anime.score:.2f}")
                    else:
                        st.metric("Score", "N/A")
                
                with col3:
                    st.write(f"**Type:** {anime.type or 'Unknown'}")
                    st.write(f"**Episodes:** {anime.episodes or 'Unknown'}")
                    st.write(f"**Year:** {anime.year or 'Unknown'}")
                
                # Expandable details section
                with st.expander(f"📖 View Details - {anime.title}", expanded=False):
                    detail_col1, detail_col2 = st.columns([1, 2])
                    
                    with detail_col1:
                        # Display image if available
                        if hasattr(anime, 'image_url') and anime.image_url:
                            st.image(anime.image_url, width=200, caption=anime.title)
                        else:
                            st.info("No image available")
                        
                        # Basic info
                        st.markdown("**📊 Information:**")
                        info_data = {
                            "MAL ID": anime.mal_id,
                            "Type": anime.type or "Unknown",
                            "Episodes": anime.episodes or "Unknown", 
                            "Status": getattr(anime, 'status', 'Unknown'),
                            "Year": anime.year or "Unknown",
                            "Season": getattr(anime, 'season', 'Unknown'),
                            "Source": getattr(anime, 'source', 'Unknown'),
                            "Rating": getattr(anime, 'rating', 'Unknown')
                        }
                        
                        for key, value in info_data.items():
                            if value and value != "Unknown":
                                st.write(f"**{key}:** {value}")
                    
                    with detail_col2:
                        # Synopsis
                        if anime.synopsis:
                            st.markdown("**📝 Synopsis:**")
                            st.write(anime.synopsis)
                        else:
                            st.info("No synopsis available")
                        
                        # Genres
                        if anime.genres:
                            st.markdown("**🎭 Genres:**")
                            genre_names = [g.name for g in anime.genres]
                            st.write(", ".join(genre_names))
                        
                        # Studios
                        if anime.studios:
                            st.markdown("**🏢 Studios:**")
                            studio_names = [s.name for s in anime.studios]
                            st.write(", ".join(studio_names))
                        
                        # Statistics
                        if hasattr(anime, 'members') or hasattr(anime, 'favorites'):
                            st.markdown("**📈 Statistics:**")
                            if hasattr(anime, 'members') and anime.members:
                                st.write(f"**Members:** {anime.members:,}")
                            if hasattr(anime, 'favorites') and anime.favorites:
                                st.write(f"**Favorites:** {anime.favorites:,}")
                            if hasattr(anime, 'popularity') and anime.popularity:
                                st.write(f"**Popularity Rank:** #{anime.popularity}")
                            if hasattr(anime, 'rank') and anime.rank:
                                st.write(f"**Overall Rank:** #{anime.rank}")
                        
                        # External link
                        if anime.mal_id:
                            st.markdown(f"[🔗 View on MyAnimeList](https://myanimelist.net/anime/{anime.mal_id})")
                
                st.divider()  # Add separator between anime entries


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
        """Render character search results with detailed information"""
        if not results:
            st.info("No characters found matching your criteria.")
            return
        
        st.success(f"Found {len(results)} character-anime relationships")
        
        # Group results by character to avoid duplicates
        character_groups = {}
        for char_name, image_url, role, anime_title, anime_score, char_mal_id, anime_mal_id in results:
            if char_name not in character_groups:
                character_groups[char_name] = {
                    'image_url': image_url,
                    'mal_id': char_mal_id,
                    'appearances': []
                }
            character_groups[char_name]['appearances'].append({
                'anime_title': anime_title,
                'role': role,
                'anime_score': anime_score,
                'anime_mal_id': anime_mal_id
            })
        
        # Display characters with expandable details
        for char_name, char_data in character_groups.items():
            with st.container():
                # Character header
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"### 👤 {char_name}")
                    appearances_count = len(char_data['appearances'])
                    st.write(f"Appears in {appearances_count} anime")
                
                with col2:
                    avg_score = sum([app['anime_score'] for app in char_data['appearances'] if app['anime_score']]) / len([app for app in char_data['appearances'] if app['anime_score']])
                    if avg_score:
                        st.metric("Avg Anime Score", f"{avg_score:.2f}")
                
                # Expandable character details
                with st.expander(f"👁️ View Character Details - {char_name}", expanded=False):
                    detail_col1, detail_col2 = st.columns([1, 2])
                    
                    with detail_col1:
                        # Character image
                        if char_data['image_url']:
                            st.image(char_data['image_url'], width=200, caption=char_name)
                        else:
                            st.info("No character image available")
                        
                        # Character info
                        if char_data['mal_id']:
                            st.write(f"**MAL ID:** {char_data['mal_id']}")
                            st.markdown(f"[🔗 View on MyAnimeList](https://myanimelist.net/character/{char_data['mal_id']})")
                    
                    with detail_col2:
                        # Anime appearances
                        st.markdown("**🎬 Anime Appearances:**")
                        
                        for appearance in char_data['appearances']:
                            with st.container():
                                app_col1, app_col2, app_col3 = st.columns([2, 1, 1])
                                
                                with app_col1:
                                    st.write(f"**{appearance['anime_title']}**")
                                
                                with app_col2:
                                    role_emoji = "⭐" if appearance['role'] == "Main" else "🎭"
                                    st.write(f"{role_emoji} {appearance['role'] or 'Unknown'}")
                                
                                with app_col3:
                                    if appearance['anime_score']:
                                        st.write(f"⭐ {appearance['anime_score']:.1f}")
                                    else:
                                        st.write("No score")
                                
                                # Link to anime
                                if appearance['anime_mal_id']:
                                    st.caption(f"[🔗 View Anime](https://myanimelist.net/anime/{appearance['anime_mal_id']})")
                                
                                st.write("---")
                
                st.divider()  # Separator between characters


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