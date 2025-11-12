# MyAnimeList Database Project

A comprehensive database-focused project for a database class, featuring a well-designed PostgreSQL schema populated with real MyAnimeList data via the Jikan API, enhanced with Hugging Face machine learning capabilities, and visualized through a Streamlit web interface.

## Project Overview

This project demonstrates:
- **Database Design**: Comprehensive relational database schema with 8 tables and proper relationships
- **Data Ingestion**: Automated pipeline using Jikan API (unofficial MyAnimeList API)
- **Machine Learning Integration**: Hugging Face models for text classification and sentiment analysis
- **Data Visualization**: Interactive Streamlit dashboard for database exploration
- **Search Functionality**: Advanced search and filtering capabilities

## Quick Start

- **Running Locally**: See [LOCAL_SETUP.md](LOCAL_SETUP.md) for detailed instructions on setting up the project on your computer

## Database Schema

### Tables

1. **anime** - Main anime information
   - Core fields: title, type, episodes, score, synopsis
   - Indexed on: mal_id, title, score, rank, popularity, year

2. **genres** - Anime genre lookup table
   - Fields: mal_id, name
   - Many-to-many relationship with anime

3. **studios** - Animation studio information
   - Fields: mal_id, name
   - Many-to-many relationship with anime

4. **themes** - Anime themes
   - Fields: mal_id, name
   - Many-to-many relationship with anime

5. **characters** - Anime characters
   - Fields: mal_id, name, image_url
   - Many-to-many relationship with anime (characters can appear in multiple anime)

6. **reviews** - User reviews
   - Fields: username, score, review_text, sentiment_score
   - One-to-many relationship with anime

7. **recommendations** - Anime recommendation network
   - Fields: anime_id, recommended_anime_id, votes
   - Self-referential relationship on anime table

8. **ml_features** - Machine learning features
   - Fields: synopsis_category, synopsis_embedding, predicted_rating
   - One-to-one relationship with anime

### Junction Tables

- **anime_genres** - Links anime to genres
- **anime_studios** - Links anime to studios
- **anime_themes** - Links anime to themes
- **anime_characters** - Links anime to characters (with role information)

## Features

### 1. Database Overview
- Real-time table statistics
- Row counts visualization
- Database initialization controls

### 2. Database Schema Viewer
- Complete schema documentation
- Table relationships diagram
- Column details with types and constraints
- Foreign key relationships
- Index information

### 3. Data Ingestion
- **Quick Ingestion**: Automated full database population
- **Custom Ingestion**: Granular control over data import
- Fetches from Jikan API:
  - Top anime (ranked by score)
  - Genres and themes
  - Characters
  - Recommendations

### 4. Search Interface
- Search by anime title (contains, starts with, exact match)
- Filter by genre, type, and minimum score
- Detailed anime information display
- Image previews

### 5. Data Explorer
- **Anime Table**: Browse all anime entries
- **Genre Analysis**: Distribution charts and statistics
- **Studio Analysis**: Top studios by anime count
- **Characters**: Character database browser
- **ML Features**: View generated predictions

### 6. ML Features (Hugging Face)
- **Synopsis Classification**: Categorizes anime into genres using BART model
- **Sentiment Analysis**: Analyzes review sentiment using DistilBERT
- **Text Embeddings**: Generates synopsis embeddings for similarity search
- Batch processing capabilities
- Progress tracking

### 7. Analytics Dashboard
- Score distribution histogram
- Temporal analysis (anime by year)
- Recommendation network analysis
- Top recommended anime visualization

## Technology Stack

### Backend
- **Python 3.11+**
- **SQLite** - Local relational database (for development/demo)
- **SQLAlchemy 2.0** - ORM and database toolkit
- **Built-in database** - No external database installation required

### APIs & Data Sources
- **Jikan API v4** - MyAnimeList data (open-source, no key required)
- **Hugging Face Inference API** - Machine learning models

### ML Models (via Hugging Face)
- `facebook/bart-large-mnli` - Text classification
- `distilbert-base-uncased-finetuned-sst-2-english` - Sentiment analysis
- `sentence-transformers/all-MiniLM-L6-v2` - Text embeddings

### Frontend & Visualization
- **Streamlit** - Web application framework
- **Plotly** - Interactive charts and graphs
- **Pandas** - Data manipulation and display

## Installation & Setup

### Prerequisites
- Python 3.11+
- Internet connection (for API calls)

### Database Setup
This project uses **SQLite** for easy setup and portability:
- **No database installation required** - SQLite is included with Python
- **Database file**: `anime_db.sqlite` (automatically created)
- **Ready to run** - No configuration needed

### Environment Variables
**Optional Environment Variables:**
- `HUGGINGFACE_TOKEN` or `HF_TOKEN` - Hugging Face API token for ML features
  - The ML features will work with the free Hugging Face Inference API
  - Providing a token improves rate limits and reliability
  - Get a free token at https://huggingface.co/settings/tokens
  - If not provided, ML features will still attempt to work but may have rate limits

### Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt.local
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Access the application at `http://localhost:8501`

### What Happens on First Run
- SQLite database (`anime_db.sqlite`) is automatically created
- Database tables are initialized
- Application is ready for data ingestion
- **No manual database setup required!**

## Usage Guide

### First-Time Setup

1. **Initialize Database**
   - Navigate to "Database Overview"
   - Click "Initialize Database" button
   - Wait for confirmation

2. **Ingest Data**
   - Go to "Data Ingestion"
   - Choose number of pages (25 anime per page)
   - Click "Run Full Ingestion"
   - Wait for completion (includes genres, anime, characters, recommendations)

3. **Generate ML Features** (Optional)
   - Navigate to "ML Features"
   - Select batch size
   - Click "Generate ML Features"
   - Wait for processing to complete

### Exploring the Database

- **Search**: Use the search interface to find specific anime
- **Schema**: Review the complete database structure
- **Data Explorer**: Browse tables and view statistics
- **Analytics**: Visualize trends and distributions

## API Rate Limiting

The Jikan API has rate limits:
- 1 request per second
- Built-in delays in ingestion pipeline
- Automatic retry handling

## Database Design Highlights

### SQLite Advantages
- **Zero configuration** - Works out of the box
- **Portable** - Single file database
- **Perfect for development** and demonstration
- **Full SQL support** - All relational features work
- **Production ready** - Handles thousands of records efficiently

### Normalization
- Proper 3NF normalization
- Separate lookup tables for genres, studios, themes
- Junction tables for many-to-many relationships

### Indexing Strategy
- Primary keys on all tables
- Foreign key indexes for join optimization
- Additional indexes on frequently queried columns (title, score, year)

### Relationships
- One-to-many: anime → characters, reviews
- Many-to-many: anime ↔ genres, studios, themes
- Self-referential: anime → recommendations → anime

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── models.py              # SQLAlchemy database models
├── database.py            # Database connection and utilities
├── jikan_ingestion.py     # Jikan API data ingestion
├── ml_features.py         # Hugging Face ML integration
├── requirements.txt.local # Python dependencies
├── anime_db.sqlite        # SQLite database file (auto-created)
├── .streamlit/            # Streamlit configuration
└── README.md              # This file
```

## Academic Value

This project demonstrates proficiency in:
- Relational database design and normalization
- SQL and ORM usage (SQLAlchemy)
- API integration and data ingestion
- Data visualization and analytics
- Machine learning integration with databases
- Full-stack application development

## Future Enhancements

- Advanced recommendation algorithms using embeddings
- User authentication and personalized watchlists
- Real-time data synchronization with MyAnimeList
- Query performance optimization and explain plans
- Database migrations with version control
- Advanced analytics (seasonal trends, genre evolution)

## Project Development Summary

This project represents a complete restart from scratch after getting lost in a previous database attempt. Starting fresh allowed for:

**Clean Architecture**: Built with modern Python practices using SQLAlchemy 2.0, Streamlit, and proper separation of concerns across modules.

**Database-First Approach**: Focusing on visualizing the database structure and relationships instead of frontend functionality helped clarify the data model and made the complex schema more understandable.

**Comprehensive Data Pipeline**: Implemented full data ingestion from MyAnimeList via Jikan API, including anime metadata, characters, genres, studios, themes, and user recommendations with proper rate limiting.

**Enhanced User Experience**: Developed an interactive Streamlit web interface with real-time database statistics, advanced search capabilities, paginated data exploration, and integrated ML features.

**Professional Development Practices**: Maintained clean Git history, comprehensive documentation, proper Python packaging with pyproject.toml, and modular code organization.

**Database Design**: Created a normalized 8-table schema supporting complex relationships between anime, characters, genres, studios, and user-generated content with full referential integrity.

The restart proved beneficial, resulting in a more robust, scalable, and maintainable codebase that successfully demonstrates modern database concepts and Python development best practices.

## Recent Enhancements & New Features

Since the initial implementation, several major enhancements have been added to improve functionality and user experience:

### **🎭 Character Search System (November 11, 2025)**
- **Advanced Character Search**: Full-featured search interface for exploring character database
- **Multi-Filter Support**: Search by character name, role (Main/Supporting), and anime title
- **Smart Result Grouping**: Characters appearing in multiple anime are grouped together with all appearances
- **Visual Enhancements**: Character images, role indicators, and anime score displays
- **Real-time Statistics**: Character count, total appearances, and average appearances per character

### **🔧 Enhanced Data Ingestion (November 11, 2025)**
- **Smart Character Ingestion**: Improved system that only processes anime without existing character data
- **Automatic Progress Tracking**: Real-time status showing how many anime need character data
- **Batch Processing**: Efficient batch ingestion that automatically continues where previous batches left off
- **No Duplicate Processing**: Intelligent system prevents reprocessing the same anime
- **Progress Visualization**: Real-time progress bars and status updates during ingestion

### **📊 Advanced Data Explorer (November 11, 2025)**
- **Dual Character Views**: Toggle between "Character-Anime Relationships" and "Unique Characters Only"
- **Enhanced Pagination**: Improved pagination system with customizable page sizes (50-500 records)
- **Advanced Sorting**: Multiple sorting options including character name, anime title, role, and MAL IDs
- **Relationship Context**: Clear explanations of many-to-many relationships and character appearances
- **Comprehensive Statistics**: Detailed breakdown of database contents and relationship counts

### **🎨 User Interface Improvements (November 11, 2025)**
- **Clean Interface**: Removed Streamlit deployment buttons and unnecessary UI elements
- **Professional Styling**: Custom theme with anime-inspired colors and clean layout
- **Enhanced Navigation**: Expanded sidebar with dedicated sections for anime and character search
- **Better Information Display**: Improved tooltips, help text, and contextual information
- **Responsive Design**: Optimized layout for better user experience

### **📈 Database Expansion (November 11, 2025)**
- **Scaled Dataset**: Expanded from 500 to 1,499 anime records (3x increase)
- **Enhanced Character Coverage**: Systematic character ingestion across all anime in database
- **Comprehensive Metadata**: Added 765+ character records, 221 studios, 78 genres
- **Relationship Optimization**: 3,470+ anime-genre relationships properly normalized
- **Performance Tuning**: Efficient queries and proper indexing for fast search operations

### **📚 Documentation & Architecture (November 11, 2025)**
- **Technology Migration**: Updated documentation from PostgreSQL to SQLite implementation
- **Clean Development History**: Removed Replit traces and established professional Git workflow
- **Comprehensive README**: Accurate setup instructions and feature documentation
- **Project Restart Summary**: Documented database-first approach and architectural decisions
- **Code Quality**: Improved separation of concerns and modular code structure

**Development Timeline:**
- **Initial Implementation**: Basic anime database with 500 records
- **November 11, 2025 - Morning**: Database expansion to 1,499 anime records
- **November 11, 2025 - Afternoon**: Enhanced Data Explorer with dual character views
- **November 11, 2025 - Evening**: Character search system and smart ingestion features
- **November 11, 2025 - Documentation**: Comprehensive feature documentation and README updates

These enhancements transformed the project from a basic database demonstration into a comprehensive, production-ready anime database application with advanced search capabilities and professional user interface design.

## Credits

- **Data Source**: [MyAnimeList](https://myanimelist.net/) via [Jikan API](https://jikan.moe/)
- **ML Models**: [Hugging Face](https://huggingface.co/)
- **Database**: SQLite (built-in with Python)
- **Framework**: Streamlit

## License

This is an educational project for database class purposes.

---

**Created for Database Class - November 2025**
*Featuring 500+ anime records with characters, genres, and ML analysis*
