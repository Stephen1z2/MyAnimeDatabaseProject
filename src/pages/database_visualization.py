"""
Database Visualization page
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.components.ui_components import (
    MetricsComponent, DistributionChartComponent, TopEntitiesComponent, ErrorHandlerComponent
)
from src.services.database_service import database_service
from src.utils.ui_helpers import (
    create_bar_chart, create_pie_chart, show_warning_message, show_info_message
)


def render_database_visualization():
    """Render the Database Visualization page"""
    st.header("📊 Database Visualization")
    
    st.markdown("""
    Interactive visualizations of your anime database structure, relationships, and data patterns.
    """)
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize the database first from the Database Overview page.")
        return
    
    try:
        with database_service as db:
            # Get data for visualizations
            counts = db.get_table_statistics()
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Database Overview", 
                "🔗 Relationship Maps", 
                "📊 Data Distribution", 
                "🎯 Entity Analytics",
                "🌐 Network Graphs"
            ])
            
            with tab1:
                _render_database_overview_tab(db, counts)
            
            with tab2:
                _render_relationship_maps_tab(counts)
            
            with tab3:
                _render_data_distribution_tab(db)
            
            with tab4:
                _render_entity_analytics_tab(db)
            
            with tab5:
                _render_network_graphs_tab(db, counts)
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "creating visualizations")


def _render_database_overview_tab(db, counts):
    """Render the Database Overview tab"""
    st.subheader("Database Scale & Growth")
    
    # Database size overview
    metrics = db.get_database_metrics()
    main_metrics = {
        "Total Records": f"{metrics.get('total_records', 0):,}",
        "Main Entities": f"{metrics.get('main_entities', 0):,}",
        "Relationships": f"{metrics.get('relationships', 0):,}"
    }
    MetricsComponent.render(main_metrics, columns=3)
    
    # Table size comparison
    stats_df = pd.DataFrame([
        {"Table": k, "Records": v, "Type": "Junction" if "_" in k else "Entity"} 
        for k, v in counts.items() if v > 0
    ])
    stats_df = stats_df.sort_values("Records", ascending=True)
    
    if not stats_df.empty:
        fig = create_bar_chart(
            stats_df, 
            x="Records", 
            y="Table",
            title="Database Table Sizes",
            orientation='h',
            color_col="Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data density metrics
    st.subheader("Data Density Analysis")
    if 'genres_per_anime' in metrics:
        density_metrics = {
            "Avg Genres/Anime": f"{metrics.get('genres_per_anime', 0):.1f}",
            "Avg Studios/Anime": f"{metrics.get('studios_per_anime', 0):.1f}",
            "Avg Characters/Anime": f"{metrics.get('chars_per_anime', 0):.1f}",
            "Avg Reviews/Anime": f"{metrics.get('reviews_per_anime', 0):.1f}"
        }
        MetricsComponent.render(density_metrics, columns=4)
    else:
        show_info_message("No anime data available for density analysis.")


def _render_relationship_maps_tab(counts):
    """Render the Relationship Maps tab"""
    st.subheader("Entity Relationship Diagrams")
    
    # Create relationship mapping
    relationships_data = [
        {"From": "anime", "To": "genres", "Type": "Many-to-Many", "Via": "anime_genres", "Count": counts.get('anime_genres', 0)},
        {"From": "anime", "To": "studios", "Type": "Many-to-Many", "Via": "anime_studios", "Count": counts.get('anime_studios', 0)},
        {"From": "anime", "To": "characters", "Type": "Many-to-Many", "Via": "anime_characters", "Count": counts.get('anime_characters', 0)},
        {"From": "anime", "To": "reviews", "Type": "One-to-Many", "Via": "Direct", "Count": counts.get('reviews', 0)},
        {"From": "anime", "To": "ml_features", "Type": "One-to-One", "Via": "Direct", "Count": counts.get('ml_features', 0)},
        {"From": "anime", "To": "recommendations", "Type": "Self-Reference", "Via": "Direct", "Count": counts.get('recommendations', 0)},
    ]
    
    rel_df = pd.DataFrame(relationships_data)
    
    # Relationship details
    st.subheader("Relationship Details")
    rel_display = rel_df.copy()
    rel_display["Strength"] = rel_display["Count"].apply(
        lambda x: "●●●●●" if x > 1000 else "●●●●○" if x > 500 else "●●●○○" if x > 100 else "●●○○○" if x > 10 else "●○○○○"
    )
    st.dataframe(rel_display, use_container_width=True, hide_index=True)


def _render_data_distribution_tab(db):
    """Render the Data Distribution tab"""
    st.subheader("Data Distribution Patterns")
    
    # Get anime data for distribution analysis
    anime_data = db.get_anime_data(limit=1000)
    
    if anime_data:
        anime_df = pd.DataFrame(anime_data, columns=['Score', 'Episodes', 'Year', 'Type', 'Title'])
        anime_df = anime_df.dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            DistributionChartComponent.render_score_distribution(anime_df)
        
        with col2:
            # Episode distribution
            if not anime_df['Episodes'].isna().all():
                episode_data = anime_df[anime_df['Episodes'] <= 50]
                if not episode_data.empty:
                    import plotly.express as px
                    fig = px.box(
                        episode_data, 
                        y="Episodes",
                        title="Episode Count Distribution (≤50 episodes)",
                        color_discrete_sequence=["#4ecdc4"]
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Year distribution
        if not anime_df['Year'].isna().all():
            year_counts = anime_df['Year'].value_counts().sort_index()
            import plotly.express as px
            fig = px.line(
                x=year_counts.index, 
                y=year_counts.values,
                title="Anime Release Timeline",
                labels={"x": "Year", "y": "Number of Anime"}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Type distribution
        DistributionChartComponent.render_type_distribution(anime_df)
    
    else:
        show_info_message("No anime data available for distribution analysis. Please ingest some data first.")


def _render_entity_analytics_tab(db):
    """Render the Entity Analytics tab"""
    st.subheader("Entity-Specific Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🎭 Top Genres**")
        genre_data = db.get_top_genres(limit=10)
        TopEntitiesComponent.render_top_genres(genre_data)
    
    with col2:
        st.write("**🏢 Top Studios**")
        studio_data = db.get_top_studios(limit=10)
        TopEntitiesComponent.render_top_studios(studio_data)
    
    # Character analysis
    st.write("**👥 Character Analysis**")
    char_data = db.get_character_appearances(limit=15)
    
    if char_data:
        char_df = pd.DataFrame(char_data, columns=['Appearances', 'Character'])
        char_df = char_df[char_df['Appearances'] > 1]  # Only multi-anime characters
        
        if not char_df.empty:
            import plotly.express as px
            fig = px.scatter(
                char_df, 
                x="Character", 
                y="Appearances",
                size="Appearances",
                title="Characters Appearing in Multiple Anime",
                color="Appearances",
                color_continuous_scale="Turbo"
            )
            fig.update_xaxis(tickangle=45)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            show_info_message("No characters appear in multiple anime yet.")
    else:
        show_info_message("No character data available.")


def _render_network_graphs_tab(db, counts):
    """Render the Network Graphs tab"""
    st.subheader("Network Relationship Graphs")
    
    st.markdown("""
    **Network Analysis**: Visualizing the connections between different entities in your database.
    """)
    
    show_info_message("""
    **Recommendation Network**: Shows how anime are connected through user recommendations.
    
    **Genre Network**: Displays relationships between anime through shared genres.
    
    **Studio Network**: Connects anime produced by the same studios.
    
    **Character Network**: Links anime that share the same characters.
    """)
    
    # Show network statistics
    network_metrics = {
        "Network Nodes": f"{counts.get('anime', 0):,}",
        "Recommendation Edges": f"{counts.get('recommendations', 0):,}",
        "Genre Connections": f"{counts.get('anime_genres', 0):,}"
    }
    MetricsComponent.render(network_metrics, columns=3)
    
    # Show recommendation strength if data exists
    if counts.get('recommendations', 0) > 0:
        rec_stats = db.get_recommendation_stats()
        if rec_stats:
            rec_metrics = {
                "Avg Rec Strength": f"{rec_stats.avg_votes:.1f}" if rec_stats.avg_votes else "N/A",
                "Max Rec Strength": f"{rec_stats.max_votes}" if rec_stats.max_votes else "N/A",
                "Total Connections": f"{rec_stats.total_recs}"
            }
            MetricsComponent.render(rec_metrics, columns=3)