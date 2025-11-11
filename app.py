import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func, inspect
from database import init_database, check_tables_exist, get_table_info, get_table_counts, get_session
from models import Anime, Genre, Studio, Theme, Character, Review, Recommendation, MLFeature, AnimeCharacter
from jikan_ingestion import run_full_ingestion, ingest_top_anime, ingest_genres
from ml_features import batch_process_ml_features
import time

st.set_page_config(
    page_title="MyAnimeList Database Project",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎌 MyAnimeList Database Project")
st.markdown("### Database-focused project with Jikan API & Hugging Face ML")

if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False

sidebar_option = st.sidebar.selectbox(
    "Navigation",
    ["Database Overview", "Database Schema", "Data Ingestion", "Search Anime", "Data Explorer", "ML Features", "Analytics"]
)

if sidebar_option == "Database Overview":
    st.header("📊 Database Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Database Status")
        
        if not check_tables_exist():
            st.warning("Database not initialized")
            if st.button("Initialize Database", type="primary"):
                with st.spinner("Creating database tables..."):
                    init_database()
                    st.session_state.db_initialized = True
                    st.success("Database initialized successfully!")
                    st.rerun()
        else:
            st.success("Database is initialized and ready")
            st.session_state.db_initialized = True
            
            st.subheader("Table Statistics")
            try:
                counts = get_table_counts()
                
                stats_df = pd.DataFrame([
                    {"Table": k, "Row Count": v} 
                    for k, v in counts.items()
                ])
                stats_df = stats_df.sort_values("Row Count", ascending=False)
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                fig = px.bar(
                    stats_df, 
                    x="Table", 
                    y="Row Count",
                    title="Records per Table",
                    color="Row Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error fetching table counts: {e}")
    
    with col2:
        st.subheader("Database Information")
        
        if st.session_state.db_initialized:
            table_info = get_table_info()
            
            st.metric("Total Tables", len(table_info))
            
            total_columns = sum(len(info['columns']) for info in table_info.values())
            st.metric("Total Columns", total_columns)
            
            total_fks = sum(len(info['foreign_keys']) for info in table_info.values())
            st.metric("Total Foreign Keys", total_fks)
            
            st.subheader("Quick Actions")
            if st.button("Refresh Statistics"):
                st.rerun()

elif sidebar_option == "Database Schema":
    st.header("🗂️ Database Schema")
    
    st.markdown("""
    This database schema is designed for MyAnimeList data with the following structure:
    """)
    
    if check_tables_exist():
        table_info = get_table_info()
        
        tab1, tab2 = st.tabs(["Schema Diagram", "Table Details"])
        
        with tab1:
            st.subheader("Entity Relationship Overview")
            
            schema_description = """
            **Main Tables:**
            - 🎬 **anime**: Core anime information (title, score, episodes, etc.)
            - 🎭 **genres**: Anime genres (Action, Comedy, Drama, etc.)
            - 🏢 **studios**: Animation studios
            - 🎨 **themes**: Anime themes
            - 👥 **characters**: Anime characters
            - 📝 **reviews**: User reviews
            - 🔗 **recommendations**: Anime recommendations
            - 🤖 **ml_features**: Machine learning features
            
            **Junction Tables:**
            - anime_genres: Many-to-many relationship between anime and genres
            - anime_studios: Many-to-many relationship between anime and studios
            - anime_themes: Many-to-many relationship between anime and themes
            """
            
            st.markdown(schema_description)
            
            relationships = """
            **Key Relationships:**
            - One Anime ↔ Many Characters (many-to-many with roles)
            - One Anime → Many Reviews (one-to-many)
            - One Anime → Many Recommendations (one-to-many)
            - One Anime ↔ Many Genres (many-to-many)
            - One Anime ↔ Many Studios (many-to-many)
            - One Anime → One ML Feature (one-to-one)
            """
            
            st.info(relationships)
        
        with tab2:
            st.subheader("Table Details")
            
            for table_name in sorted(table_info.keys()):
                with st.expander(f"📋 {table_name.upper()}", expanded=False):
                    info = table_info[table_name]
                    
                    st.write("**Columns:**")
                    cols_df = pd.DataFrame([
                        {
                            "Column": col['name'],
                            "Type": str(col['type']),
                            "Nullable": col['nullable'],
                            "Default": col.get('default', 'None')
                        }
                        for col in info['columns']
                    ])
                    st.dataframe(cols_df, use_container_width=True, hide_index=True)
                    
                    if info['foreign_keys']:
                        st.write("**Foreign Keys:**")
                        fk_df = pd.DataFrame([
                            {
                                "Column": ', '.join(fk['constrained_columns']),
                                "References": f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                            }
                            for fk in info['foreign_keys']
                        ])
                        st.dataframe(fk_df, use_container_width=True, hide_index=True)
                    
                    if info['indexes']:
                        st.write("**Indexes:**")
                        idx_df = pd.DataFrame([
                            {
                                "Name": idx['name'],
                                "Columns": ', '.join(idx['column_names']),
                                "Unique": idx['unique']
                            }
                            for idx in info['indexes']
                        ])
                        st.dataframe(idx_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Please initialize the database first from the Database Overview page.")

elif sidebar_option == "Data Ingestion":
    st.header("📥 Data Ingestion from Jikan API")
    
    st.markdown("""
    This section allows you to populate the database with real anime data from the Jikan API 
    (unofficial MyAnimeList API).
    """)
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize the database first from the Database Overview page.")
    else:
        tab1, tab2 = st.tabs(["Quick Ingestion", "Custom Ingestion"])
        
        with tab1:
            st.subheader("Quick Database Population")
            
            st.info("This will fetch top anime, genres, characters, and recommendations.")
            
            num_pages = st.slider("Number of pages to fetch (25 anime per page)", 1, 5, 2)
            
            if st.button("Run Full Ingestion", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting ingestion...")
                progress_bar.progress(10)
                
                with st.spinner("Ingesting data from Jikan API..."):
                    results = run_full_ingestion(num_pages=num_pages)
                    progress_bar.progress(100)
                    st.session_state.data_ingested = True
                
                status_text.empty()
                progress_bar.empty()
                
                st.success("Data ingestion completed!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Genres", results['genres'])
                col2.metric("Anime", results['anime'])
                col3.metric("Characters", results['characters'])
                col4.metric("Recommendations", results['recommendations'])
                
                st.balloons()
        
        with tab2:
            st.subheader("Custom Data Ingestion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Ingest Genres Only"):
                    with st.spinner("Fetching genres..."):
                        count = ingest_genres()
                    st.success(f"Ingested {count} genres!")
            
            with col2:
                page_num = st.number_input("Page number", min_value=1, max_value=100, value=1)
                if st.button("Ingest Top Anime Page"):
                    with st.spinner(f"Fetching page {page_num}..."):
                        count = ingest_top_anime(page=page_num)
                    st.success(f"Ingested {count} anime from page {page_num}!")

elif sidebar_option == "Search Anime":
    st.header("🔍 Search Anime Database")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            search_query = st.text_input("Search by anime title", placeholder="Enter anime title...")
        
        with search_col2:
            search_type = st.selectbox("Search Type", ["Contains", "Starts With", "Exact"])
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            genres = session.query(Genre).all()
            genre_names = ["All"] + [g.name for g in genres]
            selected_genre = st.selectbox("Filter by Genre", genre_names)
        
        with filter_col2:
            anime_types = ["All", "TV", "Movie", "OVA", "Special", "ONA"]
            selected_type = st.selectbox("Filter by Type", anime_types)
        
        with filter_col3:
            min_score = st.slider("Minimum Score", 0.0, 10.0, 0.0, 0.1)
        
        query = session.query(Anime)
        
        if search_query:
            if search_type == "Contains":
                query = query.filter(Anime.title.ilike(f"%{search_query}%"))
            elif search_type == "Starts With":
                query = query.filter(Anime.title.ilike(f"{search_query}%"))
            else:
                query = query.filter(Anime.title.ilike(search_query))
        
        if selected_genre != "All":
            query = query.join(Anime.genres).filter(Genre.name == selected_genre)
        
        if selected_type != "All":
            query = query.filter(Anime.type == selected_type)
        
        if min_score > 0:
            query = query.filter(Anime.score >= min_score)
        
        query = query.order_by(Anime.score.desc().nullslast())
        
        results = query.limit(50).all()
        
        st.subheader(f"Found {len(results)} anime")
        
        if results:
            for anime in results:
                with st.expander(f"⭐ {anime.title} ({anime.score or 'N/A'})"):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if anime.image_url:
                            st.image(anime.image_url, width=150)
                    
                    with col2:
                        st.write(f"**Type:** {anime.type or 'N/A'}")
                        st.write(f"**Episodes:** {anime.episodes or 'N/A'}")
                        st.write(f"**Status:** {anime.status or 'N/A'}")
                        st.write(f"**Score:** {anime.score or 'N/A'} (Scored by {anime.scored_by or 0} users)")
                        st.write(f"**Rank:** #{anime.rank or 'N/A'}")
                        
                        if anime.genres:
                            genres_str = ", ".join([g.name for g in anime.genres])
                            st.write(f"**Genres:** {genres_str}")
                        
                        if anime.studios:
                            studios_str = ", ".join([s.name for s in anime.studios])
                            st.write(f"**Studios:** {studios_str}")
                        
                        if anime.synopsis:
                            st.write(f"**Synopsis:** {anime.synopsis[:300]}...")
        else:
            st.info("No anime found matching your criteria.")
        
        session.close()

elif sidebar_option == "Data Explorer":
    st.header("📊 Data Explorer")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        explorer_tabs = st.tabs(["Anime", "Genres", "Studios", "Characters", "ML Features"])
        
        with explorer_tabs[0]:
            st.subheader("Anime Data")
            
            anime_data = session.query(
                Anime.id, Anime.title, Anime.type, Anime.episodes, 
                Anime.score, Anime.rank, Anime.status, Anime.year
            ).limit(100).all()
            
            if anime_data:
                df = pd.DataFrame(anime_data, columns=[
                    'ID', 'Title', 'Type', 'Episodes', 'Score', 'Rank', 'Status', 'Year'
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No anime data available. Please ingest data first.")
        
        with explorer_tabs[1]:
            st.subheader("Genres Analysis")
            
            genre_stats = session.query(
                Genre.name,
                func.count(anime_genres.c.anime_id).label('anime_count')
            ).join(anime_genres).group_by(Genre.name).order_by(
                func.count(anime_genres.c.anime_id).desc()
            ).all()
            
            if genre_stats:
                df = pd.DataFrame(genre_stats, columns=['Genre', 'Anime Count'])
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                
                with col2:
                    fig = px.pie(df, values='Anime Count', names='Genre', title='Anime Distribution by Genre')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No genre data available.")
        
        with explorer_tabs[2]:
            st.subheader("Studios Analysis")
            
            studio_stats = session.query(
                Studio.name,
                func.count(anime_studios.c.anime_id).label('anime_count')
            ).join(anime_studios).group_by(Studio.name).order_by(
                func.count(anime_studios.c.anime_id).desc()
            ).limit(20).all()
            
            if studio_stats:
                df = pd.DataFrame(studio_stats, columns=['Studio', 'Anime Count'])
                
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                fig = px.bar(df, x='Studio', y='Anime Count', title='Top Studios by Anime Count')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No studio data available.")
        
        with explorer_tabs[3]:
            st.subheader("Characters Data")
            
            char_data = session.query(
                Character.name, AnimeCharacter.role, Anime.title
            ).join(AnimeCharacter, Character.id == AnimeCharacter.character_id
            ).join(Anime, AnimeCharacter.anime_id == Anime.id).limit(50).all()
            
            if char_data:
                df = pd.DataFrame(char_data, columns=['Character Name', 'Role', 'Anime'])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No character data available.")
        
        with explorer_tabs[4]:
            st.subheader("ML Features")
            
            ml_data = session.query(
                Anime.title, MLFeature.synopsis_category, MLFeature.predicted_rating
            ).join(MLFeature).limit(50).all()
            
            if ml_data:
                df = pd.DataFrame(ml_data, columns=['Anime', 'Predicted Category', 'Predicted Rating'])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                category_counts = session.query(
                    MLFeature.synopsis_category,
                    func.count(MLFeature.id).label('count')
                ).group_by(MLFeature.synopsis_category).all()
                
                if category_counts:
                    cat_df = pd.DataFrame(category_counts, columns=['Category', 'Count'])
                    fig = px.bar(cat_df, x='Category', y='Count', title='ML Predicted Categories')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No ML features generated yet. Use the ML Features page to generate them.")
        
        session.close()

elif sidebar_option == "ML Features":
    st.header("🤖 Machine Learning Features")
    
    st.markdown("""
    This section uses **Hugging Face** models to add ML capabilities to the database:
    - **Synopsis Classification**: Categorize anime based on their synopsis
    - **Sentiment Analysis**: Analyze review sentiment
    - **Text Embeddings**: Generate synopsis embeddings for similarity search
    """)
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        anime_count = session.query(func.count(Anime.id)).scalar()
        ml_count = session.query(func.count(MLFeature.id)).scalar()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anime", anime_count)
        col2.metric("ML Features Generated", ml_count)
        col3.metric("Remaining", anime_count - ml_count)
        
        st.subheader("Generate ML Features")
        
        batch_size = st.slider("Number of anime to process", 1, 20, 5)
        
        st.warning("⚠️ ML processing uses Hugging Face Inference API and may take some time.")
        
        if st.button("Generate ML Features", type="primary"):
            with st.spinner(f"Processing {batch_size} anime with Hugging Face models..."):
                progress_bar = st.progress(0)
                
                results = batch_process_ml_features(limit=batch_size)
                
                progress_bar.progress(100)
            
            st.success(f"Processed {results['processed']} anime successfully!")
            if results['failed'] > 0:
                st.warning(f"{results['failed']} anime failed to process.")
            
            st.rerun()
        
        st.subheader("Recent ML Features")
        
        recent_ml = session.query(
            Anime.title, 
            MLFeature.synopsis_category,
            MLFeature.predicted_rating,
            MLFeature.created_at
        ).join(MLFeature).order_by(MLFeature.created_at.desc()).limit(10).all()
        
        if recent_ml:
            df = pd.DataFrame(recent_ml, columns=[
                'Anime', 'Predicted Category', 'Predicted Rating', 'Created At'
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No ML features generated yet.")
        
        session.close()

elif sidebar_option == "Analytics":
    st.header("📈 Database Analytics")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        tab1, tab2, tab3 = st.tabs(["Score Distribution", "Temporal Analysis", "Recommendations Network"])
        
        with tab1:
            st.subheader("Anime Score Distribution")
            
            scores = session.query(Anime.score).filter(Anime.score != None).all()
            
            if scores:
                score_list = [s[0] for s in scores]
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=score_list,
                    nbinsx=30,
                    name='Score Distribution',
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title='Anime Score Distribution',
                    xaxis_title='Score',
                    yaxis_title='Count'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                avg_score = sum(score_list) / len(score_list)
                st.metric("Average Score", f"{avg_score:.2f}")
            else:
                st.info("No score data available.")
        
        with tab2:
            st.subheader("Anime by Year")
            
            year_data = session.query(
                Anime.year,
                func.count(Anime.id).label('count')
            ).filter(Anime.year != None).group_by(Anime.year).order_by(Anime.year).all()
            
            if year_data:
                df = pd.DataFrame(year_data, columns=['Year', 'Count'])
                
                fig = px.line(df, x='Year', y='Count', title='Anime Released by Year', markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No year data available.")
        
        with tab3:
            st.subheader("Recommendation Network")
            
            rec_count = session.query(func.count(Recommendation.id)).scalar()
            st.metric("Total Recommendations", rec_count)
            
            top_recommended = session.query(
                Anime.title,
                func.count(Recommendation.id).label('recommendation_count')
            ).join(Recommendation, Recommendation.recommended_anime_id == Anime.id).group_by(
                Anime.title
            ).order_by(func.count(Recommendation.id).desc()).limit(10).all()
            
            if top_recommended:
                df = pd.DataFrame(top_recommended, columns=['Anime', 'Times Recommended'])
                
                fig = px.bar(df, x='Anime', y='Times Recommended', 
                           title='Most Recommended Anime')
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No recommendation data available.")
        
        session.close()

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**MyAnimeList Database Project**

Built with:
- PostgreSQL Database
- Jikan API (MyAnimeList)
- Hugging Face ML Models
- Streamlit Frontend
- SQLAlchemy ORM

Created for Database Class
""")
