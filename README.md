# MyAnimeList Database Project

A comprehensive database-focused project for a database class, featuring a well-designed PostgreSQL schema populated with real MyAnimeList data via the Jikan API, enhanced with Hugging Face machine learning capabilities, and visualized through a Streamlit web interface.

## Project Overview

This project demonstrates:
- **Database Design**: Comprehensive relational database schema with 8 tables and proper relationships
- **Data Ingestion**: Automated pipeline using Jikan API (unofficial MyAnimeList API)
- **Machine Learning Integration**: Hugging Face models for text classification and sentiment analysis
- **Data Visualization**: Interactive Streamlit dashboard for database exploration
- **Search Functionality**: Advanced search and filtering capabilities

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
- **Python 3.11**
- **PostgreSQL** - Relational database
- **SQLAlchemy 2.0** - ORM and database toolkit
- **psycopg2** - PostgreSQL adapter

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
- PostgreSQL database
- Internet connection (for API calls)

### Environment Variables
The following environment variables are automatically configured by Replit:
- `DATABASE_URL` - PostgreSQL connection string
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

**Optional Environment Variables:**
- `HUGGINGFACE_TOKEN` or `HF_TOKEN` - Hugging Face API token for ML features
  - The ML features will work with the free Hugging Face Inference API
  - Providing a token improves rate limits and reliability
  - Get a free token at https://huggingface.co/settings/tokens
  - If not provided, ML features will still attempt to work but may have rate limits

### Running the Application

1. Install dependencies:
```bash
uv sync
```

2. Run the Streamlit application:
```bash
streamlit run app.py --server.port 5000
```

3. Access the application at `http://localhost:5000`

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
├── pyproject.toml         # Python dependencies
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

## Credits

- **Data Source**: [MyAnimeList](https://myanimelist.net/) via [Jikan API](https://jikan.moe/)
- **ML Models**: [Hugging Face](https://huggingface.co/)
- **Database**: PostgreSQL
- **Framework**: Streamlit

## License

This is an educational project for database class purposes.

---

**Created for Database Class - November 2025**
