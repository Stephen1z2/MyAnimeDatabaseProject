import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import func, inspect
from database import init_database, check_tables_exist, get_table_info, get_table_counts, get_session
from models import Anime, Genre, Studio, Theme, Character, Review, Recommendation, MLFeature, AnimeCharacter, anime_genres, anime_studios
from jikan_ingestion import run_full_ingestion, ingest_top_anime, ingest_genres
from ml_features import batch_process_ml_features
import time

st.set_page_config(
    page_title="MyAnimeList Database Project",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's default UI elements
hide_streamlit_style = """
    <style>
    /* Hide the deploy button */
    .stDeployButton {display:none;}
    
    /* Hide the hamburger menu */
    .stMainMenu {display:none;}
    
    /* Hide "Made with Streamlit" footer */
    footer {display:none;}
    
    /* Optional: Hide the settings button */
    .stActionButton {display:none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🎌 MyAnimeList Database Project")
st.markdown("### Database-focused project with Jikan API & Hugging Face ML")

if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

if 'data_ingested' not in st.session_state:
    st.session_state.data_ingested = False

sidebar_option = st.sidebar.selectbox(
    "Navigation",
    ["Database Overview", "Database Schema", "Data Ingestion", "Search Anime", "Search Characters", "Data Explorer", "Recommendations", "Data Quality", "ML Features", "Analytics"]
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
                
                st.dataframe(stats_df, width="stretch", hide_index=True)
                
                fig = px.bar(
                    stats_df, 
                    x="Table", 
                    y="Row Count",
                    title="Records per Table",
                    color="Row Count",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, width="stretch")
                
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
                    st.dataframe(cols_df, width="stretch", hide_index=True)
                    
                    if info['foreign_keys']:
                        st.write("**Foreign Keys:**")
                        fk_df = pd.DataFrame([
                            {
                                "Column": ', '.join(fk['constrained_columns']),
                                "References": f"{fk['referred_table']}.{', '.join(fk['referred_columns'])}"
                            }
                            for fk in info['foreign_keys']
                        ])
                        st.dataframe(fk_df, width="stretch", hide_index=True)
                    
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
                        st.dataframe(idx_df, width="stretch", hide_index=True)
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
            
            # Character ingestion section
            st.subheader("Character Data Ingestion")
            
            char_col1, char_col2 = st.columns(2)
            
            # Check how many anime need characters
            session = get_session()
            anime_with_chars = session.query(func.count(func.distinct(AnimeCharacter.anime_id))).scalar()
            total_anime = session.query(func.count(Anime.id)).scalar()
            anime_without_chars = total_anime - anime_with_chars
            session.close()
            
            with char_col1:
                st.info(f"📊 **Character Status:**\n- {anime_with_chars:,} anime have characters\n- {anime_without_chars:,} anime need characters")
                
                if anime_without_chars > 0:
                    max_batch = min(100, anime_without_chars)
                    num_anime_chars = st.number_input("Number of anime to process", min_value=1, max_value=max_batch, value=min(25, max_batch), 
                                                    key="char_anime_count")
                    st.info(f"Will ingest characters for {num_anime_chars} anime **without character data**")
                else:
                    st.success("🎉 All anime already have character data!")
                
            with char_col2:
                if anime_without_chars > 0 and st.button("Ingest Characters", type="secondary"):
                    session = get_session()
                    # Get anime that DON'T have character data yet
                    anime_list = session.query(Anime).outerjoin(
                        AnimeCharacter, Anime.id == AnimeCharacter.anime_id
                    ).filter(AnimeCharacter.anime_id.is_(None)
                    ).order_by(Anime.rank.asc()).limit(num_anime_chars).all()
                    session.close()
                    
                    if not anime_list:
                        st.error("No anime found without character data!")
                    else:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_characters = 0
                        
                        for i, anime in enumerate(anime_list):
                            status_text.text(f"Processing {anime.title} ({i+1}/{len(anime_list)})")
                            progress = (i + 1) / len(anime_list)
                            progress_bar.progress(progress)
                            
                            try:
                                from jikan_ingestion import ingest_anime_characters
                                count = ingest_anime_characters(anime.mal_id)
                                total_characters += count
                                
                                # Small delay for API rate limiting
                                import time
                                time.sleep(1.2)
                                
                            except Exception as e:
                                st.warning(f"Error processing {anime.title}: {e}")
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if total_characters > 0:
                            st.success(f"Successfully ingested {total_characters} characters from {len(anime_list)} anime!")
                        else:
                            st.warning("No characters were ingested. Check if anime data exists.")

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

elif sidebar_option == "Search Characters":
    st.header("🎭 Search Character Database")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
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
        if char_search_query:
            if char_search_type == "Contains":
                query = query.filter(Character.name.ilike(f"%{char_search_query}%"))
            elif char_search_type == "Starts With":
                query = query.filter(Character.name.ilike(f"{char_search_query}%"))
            else:
                query = query.filter(Character.name.ilike(char_search_query))
        
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
                        if main_char.image_url:
                            try:
                                st.image(main_char.image_url, width=150)
                            except:
                                st.write("📷 Image unavailable")
                        else:
                            st.write("📷 No image")
                        
                        st.write(f"**MAL ID:** {main_char.mal_id}")
                    
                    with char_col2:
                        st.write(f"**Character:** {char_name}")
                        st.write(f"**Appears in {len(appearances)} anime:**")
                        
                        # Show all anime appearances
                        for appearance in appearances:
                            role_emoji = "⭐" if appearance.role == "Main" else "👥" if appearance.role == "Supporting" else "🎭"
                            score_text = f" (Score: {appearance.anime_score:.1f})" if appearance.anime_score else ""
                            st.write(f"  {role_emoji} **{appearance.anime_title}** - {appearance.role}{score_text}")
        else:
            st.info("No characters found matching your criteria.")
        
        # Show some helpful stats
        st.markdown("---")
        total_chars = session.query(func.count(Character.id)).scalar()
        total_appearances = session.query(func.count(AnimeCharacter.anime_id)).scalar()
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        stat_col1.metric("Total Characters", f"{total_chars:,}")
        stat_col2.metric("Total Character Appearances", f"{total_appearances:,}")
        stat_col3.metric("Avg Appearances per Character", f"{total_appearances/total_chars:.1f}" if total_chars > 0 else "0")
        
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
            
            # Add controls for pagination and filtering
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                records_per_page = st.selectbox(
                    "Records per page", 
                    [50, 100, 250, 500, "All"], 
                    index=1
                )
            
            with col2:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Title", "Score", "Rank", "Year", "Episodes"],
                    index=1  # Default to Score
                )
            
            with col3:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"])
            
            # Build the query
            query = session.query(
                Anime.id, Anime.title, Anime.type, Anime.episodes, 
                Anime.score, Anime.rank, Anime.status, Anime.year
            )
            
            # Apply sorting
            sort_column = {
                "Title": Anime.title,
                "Score": Anime.score, 
                "Rank": Anime.rank,
                "Year": Anime.year,
                "Episodes": Anime.episodes
            }[sort_by]
            
            if sort_order == "Descending":
                query = query.order_by(sort_column.desc())
            else:
                query = query.order_by(sort_column.asc())
            
            # Apply limit
            if records_per_page != "All":
                query = query.limit(records_per_page)
            
            anime_data = query.all()
            
            if anime_data:
                df = pd.DataFrame(anime_data, columns=[
                    'ID', 'Title', 'Type', 'Episodes', 'Score', 'Rank', 'Status', 'Year'
                ])
                
                st.info(f"Showing {len(anime_data)} anime records")
                st.dataframe(df, width='stretch', hide_index=True)
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
                    st.dataframe(df, width="stretch", hide_index=True)
                
                with col2:
                    fig = px.pie(df, values='Anime Count', names='Genre', title='Anime Distribution by Genre')
                    st.plotly_chart(fig, width="stretch")
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
                
                st.dataframe(df, width="stretch", hide_index=True)
                
                fig = px.bar(df, x='Studio', y='Anime Count', title='Top Studios by Anime Count')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No studio data available.")
        
        with explorer_tabs[3]:
            st.subheader("Characters Data")
            
            # Character view options
            view_mode = st.radio(
                "View Mode:", 
                ["Character-Anime Relationships", "Unique Characters Only"], 
                help="Character-Anime shows each character once per anime they appear in. Unique Characters shows each character only once."
            )
            
            if view_mode == "Unique Characters Only":
                # Show unique characters grouped
                unique_chars = session.query(
                    Character.name,
                    Character.mal_id,
                    func.count(func.distinct(Anime.id)).label('anime_count'),
                    func.group_concat(func.distinct(Anime.title), ' | ').label('anime_list')
                ).join(AnimeCharacter, Character.id == AnimeCharacter.character_id
                ).join(Anime, AnimeCharacter.anime_id == Anime.id
                ).group_by(Character.id, Character.name, Character.mal_id
                ).order_by(Character.name.asc()).limit(200).all()
                
                if unique_chars:
                    df = pd.DataFrame(unique_chars, columns=['Character Name', 'MAL Character ID', 'Appears in # Anime', 'Anime List'])
                    st.caption(f"Showing {len(unique_chars)} unique characters (limited to 200 for performance)")
                    st.info("💡 **Tip**: Characters appearing in multiple anime are often the same character across different seasons or movies.")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("No character data available.")
                    
            else:
                # Original character-anime relationship view
                char_per_page = st.selectbox("Characters per page:", [50, 100, 200, 500], index=0, key="char_per_page")
                
                # Get total character-anime relationships count
                total_chars = session.query(func.count(AnimeCharacter.anime_id)).scalar()
                total_char_pages = (total_chars + char_per_page - 1) // char_per_page
                
                if total_chars > 0:
                    char_page = st.selectbox(f"Page (1 to {total_char_pages}):", 
                                           range(1, total_char_pages + 1), key="char_page")
                    offset = (char_page - 1) * char_per_page
                    
                    # Character sorting options
                    char_sort_options = {
                        "Character Name (A-Z)": Character.name.asc(),
                        "Character Name (Z-A)": Character.name.desc(),
                        "Anime Title (A-Z)": Anime.title.asc(),
                        "Anime Title (Z-A)": Anime.title.desc(),
                        "Role": AnimeCharacter.role.asc(),
                        "MAL Character ID": Character.mal_id.asc()
                    }
                    
                    char_sort_by = st.selectbox("Sort by:", list(char_sort_options.keys()), key="char_sort")
                    
                    char_data = session.query(
                        Character.name, 
                        AnimeCharacter.role, 
                        Anime.title,
                        Character.mal_id,
                        Character.id
                    ).join(AnimeCharacter, Character.id == AnimeCharacter.character_id
                    ).join(Anime, AnimeCharacter.anime_id == Anime.id
                    ).order_by(char_sort_options[char_sort_by]
                    ).offset(offset).limit(char_per_page).all()
                    
                    if char_data:
                        df = pd.DataFrame(char_data, columns=['Character Name', 'Role', 'Anime', 'MAL Character ID', 'DB ID'])
                        
                        # Display pagination info and helpful note
                        st.caption(f"Showing {len(char_data)} character-anime relationships (Page {char_page} of {total_char_pages}) - Total: {total_chars}")
                        st.info("💡 **Note**: Same character appearing multiple times means they appear in multiple anime (often different seasons). Characters with the same name but different MAL IDs are different characters.")
                        
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No character data found on this page.")
                else:
                    st.info("No character data available. Use the character ingestion feature to add character data.")
        
        with explorer_tabs[4]:
            st.subheader("ML Features")
            
            ml_data = session.query(
                Anime.title, MLFeature.synopsis_category, MLFeature.predicted_rating
            ).join(MLFeature).limit(50).all()
            
            if ml_data:
                df = pd.DataFrame(ml_data, columns=['Anime', 'Predicted Category', 'Predicted Rating'])
                st.dataframe(df, width="stretch", hide_index=True)
                
                category_counts = session.query(
                    MLFeature.synopsis_category,
                    func.count(MLFeature.id).label('count')
                ).group_by(MLFeature.synopsis_category).all()
                
                if category_counts:
                    cat_df = pd.DataFrame(category_counts, columns=['Category', 'Count'])
                    fig = px.bar(cat_df, x='Category', y='Count', title='ML Predicted Categories')
                    st.plotly_chart(fig, width="stretch")
            else:
                st.info("No ML features generated yet. Use the ML Features page to generate them.")
        
        session.close()

elif sidebar_option == "Recommendations":
    st.header("🎯 Anime Recommendations")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        st.markdown("""
        **Find anime similar to ones you love!** This recommendation engine analyzes:
        - **Shared Genres**: Anime with overlapping genres
        - **Same Studio**: Other works from the same studio
        - **Similar Scores**: Anime with comparable ratings
        """)
        
        # Get all anime for selection
        all_anime = session.query(Anime.title, Anime.mal_id, Anime.score, Anime.image_url).order_by(Anime.title).all()
        anime_options = {f"{title} (Score: {score if score else 'N/A'})": mal_id 
                        for title, mal_id, score, _ in all_anime}
        
        # Anime selection
        st.subheader("🎬 Select an Anime")
        selected_anime_display = st.selectbox(
            "Choose an anime to get recommendations:",
            options=list(anime_options.keys()),
            key="rec_anime_select"
        )
        
        if selected_anime_display:
            selected_mal_id = anime_options[selected_anime_display]
            
            # Get the selected anime details
            selected_anime = session.query(Anime).filter(Anime.mal_id == selected_mal_id).first()
            
            if selected_anime:
                # Display selected anime info
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if selected_anime.image_url:
                        st.image(selected_anime.image_url, width=150)
                    else:
                        st.write("🖼️ No image")
                
                with col2:
                    st.markdown(f"### {selected_anime.title}")
                    st.write(f"**Score:** {selected_anime.score if selected_anime.score else 'N/A'}")
                    st.write(f"**Episodes:** {selected_anime.episodes if selected_anime.episodes else 'N/A'}")
                    st.write(f"**Status:** {selected_anime.status if selected_anime.status else 'N/A'}")
                    
                    # Get genres
                    genres = [g.name for g in selected_anime.genres]
                    st.write(f"**Genres:** {', '.join(genres) if genres else 'N/A'}")
                    
                    # Get studio
                    studios = [s.name for s in selected_anime.studios]
                    st.write(f"**Studios:** {', '.join(studios) if studios else 'N/A'}")
                
                st.markdown("---")
                
                # Recommendation Settings
                st.subheader("⚙️ Recommendation Settings")
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    genre_weight = st.slider("Genre Similarity Weight", 0.0, 1.0, 0.6, 0.1)
                
                with rec_col2:
                    studio_weight = st.slider("Studio Match Weight", 0.0, 1.0, 0.3, 0.1)
                
                with rec_col3:
                    score_weight = st.slider("Score Similarity Weight", 0.0, 1.0, 0.1, 0.1)
                
                # Score range settings
                score_range = st.slider("Score Range (±)", 0.5, 3.0, 1.0, 0.5)
                max_results = st.slider("Maximum Results", 5, 20, 10)
                
                if st.button("🎯 Get Recommendations", type="primary"):
                    with st.spinner("Analyzing anime similarities..."):
                        
                        # Get all other anime (excluding the selected one)
                        other_anime = session.query(Anime).filter(Anime.mal_id != selected_mal_id).all()
                        
                        recommendations = []
                        selected_genres = set(g.name for g in selected_anime.genres)
                        selected_studios = set(s.name for s in selected_anime.studios)
                        selected_score = selected_anime.score or 0
                        
                        for anime in other_anime:
                            similarity_score = 0
                            reasons = []
                            
                            # Genre similarity
                            anime_genres = set(g.name for g in anime.genres)
                            if selected_genres and anime_genres:
                                genre_overlap = len(selected_genres.intersection(anime_genres))
                                genre_similarity = genre_overlap / len(selected_genres.union(anime_genres))
                                similarity_score += genre_similarity * genre_weight
                                
                                if genre_overlap > 0:
                                    shared_genres = selected_genres.intersection(anime_genres)
                                    reasons.append(f"Shared genres: {', '.join(shared_genres)}")
                            
                            # Studio similarity
                            anime_studios = set(s.name for s in anime.studios)
                            if selected_studios and anime_studios:
                                studio_overlap = len(selected_studios.intersection(anime_studios))
                                if studio_overlap > 0:
                                    similarity_score += studio_weight
                                    shared_studios = selected_studios.intersection(anime_studios)
                                    reasons.append(f"Same studio: {', '.join(shared_studios)}")
                            
                            # Score similarity
                            if selected_score > 0 and anime.score:
                                score_diff = abs(selected_score - anime.score)
                                if score_diff <= score_range:
                                    score_similarity = 1 - (score_diff / score_range)
                                    similarity_score += score_similarity * score_weight
                                    reasons.append(f"Similar score: {anime.score:.1f} vs {selected_score:.1f}")
                            
                            # Only include if there's some similarity
                            if similarity_score > 0:
                                recommendations.append({
                                    'anime': anime,
                                    'score': similarity_score,
                                    'reasons': reasons
                                })
                        
                        # Sort by similarity score and get top results
                        recommendations.sort(key=lambda x: x['score'], reverse=True)
                        top_recommendations = recommendations[:max_results]
                        
                        if top_recommendations:
                            st.success(f"🎉 Found {len(top_recommendations)} recommendations!")
                            
                            # Display recommendations
                            for i, rec in enumerate(top_recommendations, 1):
                                anime = rec['anime']
                                score = rec['score']
                                reasons = rec['reasons']
                                
                                with st.container():
                                    rec_col1, rec_col2 = st.columns([1, 4])
                                    
                                    with rec_col1:
                                        if anime.image_url:
                                            st.image(anime.image_url, width=100)
                                        else:
                                            st.write("🖼️")
                                    
                                    with rec_col2:
                                        st.markdown(f"#### {i}. {anime.title}")
                                        st.write(f"**Similarity Score:** {score:.2f}")
                                        st.write(f"**Score:** {anime.score if anime.score else 'N/A'}")
                                        st.write(f"**Episodes:** {anime.episodes if anime.episodes else 'N/A'}")
                                        
                                        # Show genres
                                        rec_genres = [g.name for g in anime.genres]
                                        if rec_genres:
                                            st.write(f"**Genres:** {', '.join(rec_genres)}")
                                        
                                        # Show reasons
                                        if reasons:
                                            st.write(f"**Why recommended:** {' • '.join(reasons)}")
                                        
                                        # Synopsis preview
                                        if anime.synopsis:
                                            synopsis_preview = anime.synopsis[:200] + "..." if len(anime.synopsis) > 200 else anime.synopsis
                                            st.write(f"**Synopsis:** {synopsis_preview}")
                                
                                st.markdown("---")
                        else:
                            st.warning("😔 No similar anime found with current settings. Try adjusting the weights or score range.")
        
        session.close()

elif sidebar_option == "Data Quality":
    st.header("🔍 Data Quality & Integrity")
    
    if not st.session_state.db_initialized:
        st.warning("Please initialize and populate the database first.")
    else:
        session = get_session()
        
        # Data Quality Overview
        st.subheader("📊 Data Quality Overview")
        
        # Get basic statistics
        anime_count = session.query(func.count(Anime.id)).scalar()
        char_count = session.query(func.count(Character.id)).scalar()
        genre_count = session.query(func.count(Genre.id)).scalar()
        studio_count = session.query(func.count(Studio.id)).scalar()
        
        # Quality metrics
        anime_with_synopsis = session.query(func.count(Anime.id)).filter(Anime.synopsis.isnot(None), Anime.synopsis != "").scalar()
        anime_with_scores = session.query(func.count(Anime.id)).filter(Anime.score.isnot(None)).scalar()
        anime_with_images = session.query(func.count(Anime.id)).filter(Anime.image_url.isnot(None), Anime.image_url != "").scalar()
        
        # Display quality metrics
        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
        qual_col1.metric("Total Anime", f"{anime_count:,}")
        qual_col2.metric("With Synopsis", f"{anime_with_synopsis:,}", f"{(anime_with_synopsis/anime_count*100):.1f}%" if anime_count > 0 else "0%")
        qual_col3.metric("With Scores", f"{anime_with_scores:,}", f"{(anime_with_scores/anime_count*100):.1f}%" if anime_count > 0 else "0%")
        qual_col4.metric("With Images", f"{anime_with_images:,}", f"{(anime_with_images/anime_count*100):.1f}%" if anime_count > 0 else "0%")
        
        st.markdown("---")
        
        # Duplicate Detection Tabs
        quality_tabs = st.tabs(["🔍 Duplicate Detection", "📋 Missing Data", "🔗 Orphaned Records", "📈 Data Consistency"])
        
        with quality_tabs[0]:
            st.subheader("Duplicate Detection")
            
            detect_col1, detect_col2 = st.columns(2)
            
            with detect_col1:
                st.markdown("#### 🎬 Anime Duplicates")
                
                # Find potential anime duplicates by title similarity
                anime_duplicates = session.query(
                    Anime.title,
                    func.count(Anime.id).label('count'),
                    func.group_concat(Anime.mal_id).label('mal_ids')
                ).group_by(Anime.title).having(func.count(Anime.id) > 1).order_by(func.count(Anime.id).desc()).all()
                
                if anime_duplicates:
                    st.warning(f"Found {len(anime_duplicates)} potential anime duplicates:")
                    for title, count, mal_ids in anime_duplicates[:10]:  # Show top 10
                        st.write(f"**{title}** - {count} entries (MAL IDs: {mal_ids})")
                else:
                    st.success("✅ No exact title duplicates found")
                
                # Check for similar titles (basic similarity)
                if st.button("Check Similar Titles", key="anime_similar"):
                    with st.spinner("Analyzing title similarities..."):
                        all_anime = session.query(Anime.title, Anime.mal_id).all()
                        similar_pairs = []
                        
                        for i, (title1, mal_id1) in enumerate(all_anime):
                            for title2, mal_id2 in all_anime[i+1:i+20]:  # Limit for performance
                                # Simple similarity check
                                if title1.lower() in title2.lower() or title2.lower() in title1.lower():
                                    if title1 != title2:
                                        similar_pairs.append((title1, title2, mal_id1, mal_id2))
                        
                        if similar_pairs:
                            st.warning(f"Found {len(similar_pairs)} potentially similar titles:")
                            for title1, title2, mal_id1, mal_id2 in similar_pairs[:5]:
                                st.write(f"• {title1} (MAL: {mal_id1})")
                                st.write(f"• {title2} (MAL: {mal_id2})")
                                st.write("---")
                        else:
                            st.success("✅ No similar titles detected")
            
            with detect_col2:
                st.markdown("#### 👥 Character Duplicates")
                
                # Find character duplicates
                char_duplicates = session.query(
                    Character.name,
                    func.count(Character.id).label('count'),
                    func.group_concat(Character.mal_id).label('mal_ids')
                ).group_by(Character.name).having(func.count(Character.id) > 1).order_by(func.count(Character.id).desc()).all()
                
                if char_duplicates:
                    st.info(f"Found {len(char_duplicates)} characters with same name:")
                    for name, count, mal_ids in char_duplicates[:10]:
                        st.write(f"**{name}** - {count} entries (MAL IDs: {mal_ids})")
                    st.caption("💡 Note: Same character name in different anime is normal (e.g., different 'Haku' characters)")
                else:
                    st.success("✅ No character name duplicates")
                
                # Check for duplicate MAL IDs (actual duplicates)
                char_mal_duplicates = session.query(
                    Character.mal_id,
                    func.count(Character.id).label('count'),
                    func.group_concat(Character.name).label('names')
                ).group_by(Character.mal_id).having(func.count(Character.id) > 1).all()
                
                if char_mal_duplicates:
                    st.error(f"🚨 Found {len(char_mal_duplicates)} actual character duplicates (same MAL ID):")
                    for mal_id, count, names in char_mal_duplicates:
                        st.write(f"**MAL ID {mal_id}** - {count} entries: {names}")
                else:
                    st.success("✅ No duplicate MAL IDs found")
        
        with quality_tabs[1]:
            st.subheader("Missing Data Analysis")
            
            # Anime missing data
            missing_synopsis = session.query(func.count(Anime.id)).filter(
                (Anime.synopsis.is_(None)) | (Anime.synopsis == "")
            ).scalar()
            missing_scores = session.query(func.count(Anime.id)).filter(Anime.score.is_(None)).scalar()
            missing_images = session.query(func.count(Anime.id)).filter(
                (Anime.image_url.is_(None)) | (Anime.image_url == "")
            ).scalar()
            missing_episodes = session.query(func.count(Anime.id)).filter(Anime.episodes.is_(None)).scalar()
            
            miss_col1, miss_col2 = st.columns(2)
            
            with miss_col1:
                st.markdown("#### 🎬 Anime Missing Data")
                st.metric("Missing Synopsis", f"{missing_synopsis:,}", f"{(missing_synopsis/anime_count*100):.1f}%" if anime_count > 0 else "0%")
                st.metric("Missing Scores", f"{missing_scores:,}", f"{(missing_scores/anime_count*100):.1f}%" if anime_count > 0 else "0%")
                st.metric("Missing Images", f"{missing_images:,}", f"{(missing_images/anime_count*100):.1f}%" if anime_count > 0 else "0%")
                st.metric("Missing Episodes", f"{missing_episodes:,}", f"{(missing_episodes/anime_count*100):.1f}%" if anime_count > 0 else "0%")
            
            with miss_col2:
                st.markdown("#### 👥 Character Missing Data")
                char_missing_images = session.query(func.count(Character.id)).filter(
                    (Character.image_url.is_(None)) | (Character.image_url == "")
                ).scalar()
                
                role_missing = session.query(func.count(AnimeCharacter.anime_id)).filter(
                    (AnimeCharacter.role.is_(None)) | (AnimeCharacter.role == "")
                ).scalar()
                
                st.metric("Characters Missing Images", f"{char_missing_images:,}", f"{(char_missing_images/char_count*100):.1f}%" if char_count > 0 else "0%")
                st.metric("Character Roles Missing", f"{role_missing:,}")
        
        with quality_tabs[2]:
            st.subheader("Orphaned Records Detection")
            
            # Check for orphaned records
            orphan_col1, orphan_col2 = st.columns(2)
            
            with orphan_col1:
                st.markdown("#### 🔗 Relationship Integrity")
                
                # Characters without anime relationships
                orphaned_chars = session.query(func.count(Character.id)).outerjoin(
                    AnimeCharacter, Character.id == AnimeCharacter.character_id
                ).filter(AnimeCharacter.character_id.is_(None)).scalar()
                
                # Anime without genres
                anime_no_genres = session.query(func.count(Anime.id)).outerjoin(
                    anime_genres, Anime.id == anime_genres.c.anime_id
                ).filter(anime_genres.c.anime_id.is_(None)).scalar()
                
                st.metric("Orphaned Characters", f"{orphaned_chars:,}")
                st.metric("Anime Without Genres", f"{anime_no_genres:,}")
                
                if orphaned_chars > 0:
                    st.warning(f"⚠️ {orphaned_chars} characters are not linked to any anime")
                
                if anime_no_genres > 0:
                    st.warning(f"⚠️ {anime_no_genres} anime have no genre classifications")
            
            with orphan_col2:
                st.markdown("#### 📊 Reference Integrity")
                
                # Check for broken references
                total_char_relations = session.query(func.count(AnimeCharacter.anime_id)).scalar()
                total_genre_relations = session.query(func.count(anime_genres.c.anime_id)).scalar()
                
                st.metric("Character-Anime Links", f"{total_char_relations:,}")
                st.metric("Anime-Genre Links", f"{total_genre_relations:,}")
                
                # Data completeness score
                completeness_score = (
                    (anime_with_synopsis / anime_count) * 0.3 +
                    (anime_with_scores / anime_count) * 0.3 +
                    ((char_count - orphaned_chars) / char_count if char_count > 0 else 1) * 0.4
                ) * 100 if anime_count > 0 else 0
                
                st.metric("Data Completeness Score", f"{completeness_score:.1f}%")
        
        with quality_tabs[3]:
            st.subheader("Data Consistency Checks")
            
            consist_col1, consist_col2 = st.columns(2)
            
            with consist_col1:
                st.markdown("#### 📊 Score Validation")
                
                # Invalid scores
                invalid_scores = session.query(func.count(Anime.id)).filter(
                    (Anime.score < 0) | (Anime.score > 10)
                ).scalar()
                
                # Suspicious scores (very low or very high)
                very_low_scores = session.query(func.count(Anime.id)).filter(Anime.score < 2.0).scalar()
                very_high_scores = session.query(func.count(Anime.id)).filter(Anime.score > 9.5).scalar()
                
                st.metric("Invalid Scores (0-10 range)", invalid_scores)
                st.metric("Very Low Scores (<2.0)", very_low_scores)
                st.metric("Very High Scores (>9.5)", very_high_scores)
                
                # Score distribution
                avg_score = session.query(func.avg(Anime.score)).filter(Anime.score.isnot(None)).scalar()
                if avg_score:
                    st.metric("Average Score", f"{avg_score:.2f}")
            
            with consist_col2:
                st.markdown("#### 📅 Date Validation")
                
                # Future dates
                from datetime import date
                today = date.today()
                future_dates = session.query(func.count(Anime.id)).filter(Anime.aired_from > today).scalar()
                
                # Very old dates (suspicious)
                very_old = session.query(func.count(Anime.id)).filter(Anime.aired_from < date(1950, 1, 1)).scalar()
                
                st.metric("Future Dates", future_dates)
                st.metric("Very Old Dates (<1950)", very_old)
                
                # Episode count validation
                high_episodes = session.query(func.count(Anime.id)).filter(Anime.episodes > 2000).scalar()
                zero_episodes = session.query(func.count(Anime.id)).filter(Anime.episodes == 0).scalar()
                
                st.metric("High Episode Count (>2000)", high_episodes)
                st.metric("Zero Episodes", zero_episodes)
        
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
            st.dataframe(df, width="stretch", hide_index=True)
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
                st.plotly_chart(fig, width="stretch")
                
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
                st.plotly_chart(fig, width="stretch")
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
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width="stretch")
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
