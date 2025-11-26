"""
Database Schema page
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import check_tables_exist, get_table_info
from src.components.ui_components import ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message


def render_database_schema():
    """Render the Database Schema page"""
    st.header("🗂️ Database Schema")
    
    st.markdown("""
    This database schema is designed for MyAnimeList data with the following structure:
    """)
    
    if check_tables_exist():
        try:
            table_info = get_table_info()
            
            tab1, tab2 = st.tabs(["Schema Diagram", "Table Details"])
            
            with tab1:
                _render_schema_diagram()
            
            with tab2:
                _render_table_details(table_info)
        
        except Exception as e:
            ErrorHandlerComponent.handle_database_error(e, "loading schema information")
    else:
        show_warning_message("Please initialize the database first from the Database Overview page.")


def _render_schema_diagram():
    """Render the schema diagram tab"""
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


def _render_table_details(table_info):
    """Render the table details tab"""
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