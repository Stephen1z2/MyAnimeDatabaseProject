"""
MyAnimeList Database Project - Clean Refactored Version

This is the main entry point for the refactored anime database application.
The application has been restructured into a modular architecture for better
maintainability, testability, and scalability.

Original monolithic version backed up as: app_monolithic.py
"""

import streamlit as st
import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# Import the main application
try:
    from src.main import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"Error importing refactored modules: {e}")
    st.info("If you encounter import issues, try running: `streamlit run src/main.py` instead")
except Exception as e:
    st.error(f"Application error: {e}")
    st.info("Please check the console for detailed error information")