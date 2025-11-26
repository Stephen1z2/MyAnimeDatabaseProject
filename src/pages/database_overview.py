"""
Database Overview page
"""

import streamlit as st
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import init_database, check_tables_exist, get_table_info
from src.components.ui_components import DatabaseStatsComponent, MetricsComponent, ErrorHandlerComponent
from src.services.database_service import database_service
from src.utils.ui_helpers import show_success_message, show_warning_message


def render_database_overview():
    """Render the Database Overview page"""
    st.header("📊 Database Overview")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Database Status")
        
        if not check_tables_exist():
            show_warning_message("Database not initialized")
            if st.button("Initialize Database", type="primary"):
                try:
                    with st.spinner("Creating database tables..."):
                        init_database()
                        st.session_state.db_initialized = True
                        show_success_message("Database initialized successfully!")
                        st.rerun()
                except Exception as e:
                    ErrorHandlerComponent.handle_database_error(e, "database initialization")
        else:
            show_success_message("Database is initialized and ready")
            st.session_state.db_initialized = True
            
            st.subheader("Table Statistics")
            try:
                with database_service as db:
                    counts = db.get_table_statistics()
                    DatabaseStatsComponent.render(counts)
                    
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "fetching table counts")
    
    with col2:
        st.subheader("Database Information")
        
        if st.session_state.get('db_initialized', False):
            try:
                table_info = get_table_info()
                
                if table_info:
                    st.write("**Database Schema:**")
                    schema_stats = {
                        "Total Tables": len(table_info),
                        "Entity Tables": len([t for t in table_info.keys() if '_' not in t]),
                        "Junction Tables": len([t for t in table_info.keys() if '_' in t]),
                    }
                    
                    for label, value in schema_stats.items():
                        st.write(f"- {label}: {value}")
                    
                    # Show table types
                    st.write("**Table Categories:**")
                    entity_tables = [t for t in table_info.keys() if '_' not in t]
                    junction_tables = [t for t in table_info.keys() if '_' in t]
                    
                    if entity_tables:
                        st.write("*Entity Tables:*")
                        for table in sorted(entity_tables):
                            st.write(f"  • {table}")
                    
                    if junction_tables:
                        st.write("*Junction Tables:*")
                        for table in sorted(junction_tables):
                            st.write(f"  • {table}")
                
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "fetching table information")
        else:
            st.info("Initialize the database to view detailed information.")
    
    # Database metrics section
    if st.session_state.get('db_initialized', False):
        st.subheader("Database Metrics")
        try:
            with database_service as db:
                metrics = db.get_database_metrics()
                
                # Display main metrics
                main_metrics = {
                    "Total Records": f"{metrics.get('total_records', 0):,}",
                    "Main Entities": f"{metrics.get('main_entities', 0):,}",
                    "Relationships": f"{metrics.get('relationships', 0):,}"
                }
                MetricsComponent.render(main_metrics, columns=3)
                
                # Display density metrics if available
                if 'genres_per_anime' in metrics:
                    st.write("**Data Density Metrics:**")
                    density_metrics = {
                        "Avg Genres/Anime": f"{metrics.get('genres_per_anime', 0):.1f}",
                        "Avg Studios/Anime": f"{metrics.get('studios_per_anime', 0):.1f}",
                        "Avg Characters/Anime": f"{metrics.get('chars_per_anime', 0):.1f}",
                        "Avg Reviews/Anime": f"{metrics.get('reviews_per_anime', 0):.1f}"
                    }
                    MetricsComponent.render(density_metrics, columns=4)
                
        except Exception as e:
            ErrorHandlerComponent.handle_database_error(e, "fetching database metrics")