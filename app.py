"""
MyAnimeList Database Project with Enhanced ML
Complete database functionality with advanced machine learning recommendations

This app includes all original features plus enhanced ML capabilities:
- Database overview and visualization
- Anime search and data exploration  
- Hidden gems discovery using AI analysis
- Personalized anime recommendations with external data
- Enhanced ML with 10 years of MyAnimeList data
- Analytics and database statistics
"""

import streamlit as st
import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Import the streamlined main application
try:
    from src.main import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("If you encounter issues, try running: `streamlit run app_ml_only.py` for the standalone ML app")
except Exception as e:
    st.error(f"Application error: {e}")
    st.info("Please check the console for detailed error information")