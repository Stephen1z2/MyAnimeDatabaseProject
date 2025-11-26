"""
Application configuration settings
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "MyAnimeList Database Project",
    "page_icon": "🎌",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Navigation options
NAVIGATION_OPTIONS = [
    "Database Overview",
    "Database Visualization", 
    "Search Anime",
    "Search Characters",
    "💎 Hidden Gems Finder",
    "🎯 Smart Recommendations", 
    "🚀 Enhanced ML",
    "Data Explorer",
    "Analytics"
]

# Styling configuration
HIDE_STREAMLIT_STYLE = """
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

# Database configuration
DB_CONFIG = {
    "sqlite_path": "anime_db.sqlite",
    "echo": False
}

# API configuration
API_CONFIG = {
    "jikan_base_url": "https://api.jikan.moe/v4",
    "rate_limit_delay": 1.0,  # seconds between requests
    "batch_size": 25,
    "max_retries": 3
}

# ML configuration
ML_CONFIG = {
    "models_dir": "models",
    "default_test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5
}

# UI configuration
UI_CONFIG = {
    "default_page_size": 100,
    "max_page_size": 500,
    "chart_height": 400,
    "table_height": 400
}