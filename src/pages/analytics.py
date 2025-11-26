"""
Analytics page
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from sqlalchemy import func

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import get_session
from models import Anime, Genre, Studio, Recommendation, anime_genres, anime_studios
from src.components.ui_components import ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message, create_bar_chart, create_histogram


def render_analytics():
    """Render the Analytics page"""
    st.header("📈 Database Analytics")
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize and populate the database first.")
        return
    
    st.markdown("""
    Comprehensive analytics and insights from your anime database.
    """)
    
    try:
        session = get_session()
        
        tab1, tab2, tab3 = st.tabs(["📊 Score Analysis", "📅 Temporal Analysis", "🔗 Network Analysis"])
        
        with tab1:
            _render_score_analysis(session)
        
        with tab2:
            _render_temporal_analysis(session)
        
        with tab3:
            _render_network_analysis(session)
    
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "analytics")
    finally:
        if 'session' in locals():
            session.close()


def _render_score_analysis(session):
    """Render score analysis tab"""
    st.subheader("Score Distribution Analysis")
    
    # Get anime scores
    scores = session.query(Anime.score).filter(Anime.score.isnot(None)).all()
    
    if scores:
        score_values = [s[0] for s in scores]
        score_df = pd.DataFrame({'Score': score_values})
        
        # Score distribution histogram
        fig = create_histogram(score_df, 'Score', "Anime Score Distribution", nbins=30)
        st.plotly_chart(fig, use_container_width=True)
        
        # Score statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Score", f"{score_df['Score'].mean():.2f}")
        with col2:
            st.metric("Median Score", f"{score_df['Score'].median():.2f}")
        with col3:
            st.metric("Highest Score", f"{score_df['Score'].max():.2f}")
        with col4:
            st.metric("Lowest Score", f"{score_df['Score'].min():.2f}")
        
        # Score ranges
        st.subheader("Score Range Analysis")
        score_ranges = pd.cut(score_df['Score'], bins=[0, 5, 6, 7, 8, 9, 10], labels=['0-5', '5-6', '6-7', '7-8', '8-9', '9-10'])
        range_counts = score_ranges.value_counts().sort_index()
        
        range_df = pd.DataFrame({
            'Score Range': range_counts.index,
            'Count': range_counts.values
        })
        
        fig = create_bar_chart(range_df, 'Score Range', 'Count', "Anime by Score Range")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No score data available for analysis.")


def _render_temporal_analysis(session):
    """Render temporal analysis tab"""
    st.subheader("Temporal Trends")
    
    # Get anime by year
    years = session.query(Anime.year, func.count(Anime.id).label('count')).filter(
        Anime.year.isnot(None)
    ).group_by(Anime.year).order_by(Anime.year.asc()).all()
    
    if years:
        year_df = pd.DataFrame(years, columns=['Year', 'Count'])
        
        # Timeline chart
        fig = px.line(
            year_df, 
            x='Year', 
            y='Count',
            title="Anime Releases by Year",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Decade analysis
        st.subheader("Releases by Decade")
        year_df['Decade'] = (year_df['Year'] // 10) * 10
        decade_counts = year_df.groupby('Decade')['Count'].sum().reset_index()
        decade_counts['Decade Label'] = decade_counts['Decade'].astype(str) + 's'
        
        fig = create_bar_chart(
            decade_counts, 
            'Decade Label', 
            'Count', 
            "Anime Releases by Decade"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Year statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Earliest Year", int(year_df['Year'].min()))
        with col2:
            st.metric("Latest Year", int(year_df['Year'].max()))
        with col3:
            peak_year = year_df.loc[year_df['Count'].idxmax(), 'Year']
            st.metric("Peak Year", int(peak_year))
    
    else:
        st.info("No year data available for temporal analysis.")


def _render_network_analysis(session):
    """Render network analysis tab"""
    st.subheader("Recommendation Network Analysis")
    
    # Get recommendation statistics
    rec_count = session.query(func.count(Recommendation.id)).scalar()
    
    if rec_count > 0:
        # Basic network statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Recommendations", f"{rec_count:,}")
            
            # Average recommendations per anime
            anime_with_recs = session.query(
                func.count(func.distinct(Recommendation.anime_id))
            ).scalar()
            avg_recs = rec_count / anime_with_recs if anime_with_recs > 0 else 0
            st.metric("Avg Recommendations per Anime", f"{avg_recs:.1f}")
        
        with col2:
            # Most recommended anime
            top_recommended = session.query(
                Recommendation.recommended_anime_id,
                func.count(Recommendation.id).label('rec_count')
            ).group_by(
                Recommendation.recommended_anime_id
            ).order_by(
                func.count(Recommendation.id).desc()
            ).limit(5).all()
            
            if top_recommended:
                st.write("**Most Recommended Anime:**")
                for anime_id, count in top_recommended:
                    anime = session.query(Anime).filter(Anime.id == anime_id).first()
                    if anime:
                        st.write(f"• {anime.title}: {count} recommendations")
        
        # Recommendation strength analysis
        st.subheader("Recommendation Strength")
        vote_data = session.query(Recommendation.votes).filter(
            Recommendation.votes.isnot(None)
        ).all()
        
        if vote_data:
            votes = [v[0] for v in vote_data]
            vote_df = pd.DataFrame({'Votes': votes})
            
            fig = create_histogram(vote_df, 'Votes', "Recommendation Vote Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("No recommendation data available for network analysis.")
    
    # Genre co-occurrence analysis
    st.subheader("Genre Co-occurrence Analysis")
    _render_genre_cooccurrence(session)


def _render_genre_cooccurrence(session):
    """Render genre co-occurrence analysis"""
    # Get top genres for co-occurrence analysis
    top_genres = session.query(
        Genre.name,
        func.count(anime_genres.c.anime_id).label('count')
    ).join(anime_genres).group_by(Genre.name).order_by(
        func.count(anime_genres.c.anime_id).desc()
    ).limit(10).all()
    
    if len(top_genres) >= 2:
        genre_names = [g[0] for g in top_genres]
        genre_counts = [g[1] for g in top_genres]
        
        genre_df = pd.DataFrame({
            'Genre': genre_names,
            'Anime Count': genre_counts
        })
        
        fig = create_bar_chart(
            genre_df, 
            'Genre', 
            'Anime Count', 
            "Top 10 Genres by Popularity"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Insufficient genre data for co-occurrence analysis.")