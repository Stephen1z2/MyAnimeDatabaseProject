# MyAnimeList Database Project

## Overview

This is a database-focused educational project that demonstrates comprehensive database design and data engineering principles. The application ingests real anime data from MyAnimeList via the Jikan API, stores it in a PostgreSQL relational database with 8+ tables, and enhances the data using Hugging Face machine learning models for text classification and sentiment analysis. The project features an interactive Streamlit dashboard for data exploration, search functionality, and analytics visualization.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Application Framework
- **Frontend**: Streamlit web application for interactive data visualization and exploration
- **Backend**: Python-based data processing and API integration layer
- **Database**: PostgreSQL relational database with SQLAlchemy ORM

### Database Architecture

**Core Design Pattern**: Relational database with normalized schema using junction tables for many-to-many relationships.

**Main Tables**:
1. `anime` - Central table storing anime metadata (title, type, episodes, score, synopsis)
2. `genres` - Genre lookup table with many-to-many relationship to anime
3. `studios` - Animation studio information with many-to-many relationship to anime
4. `themes` - Thematic categorization with many-to-many relationship to anime
5. `characters` - Character information with many-to-many relationship to anime via `anime_characters` junction table
6. `reviews` - User reviews with one-to-many relationship to anime, includes sentiment scores
7. `recommendations` - Self-referential recommendation network showing anime-to-anime relationships
8. `ml_features` - Machine learning-generated features with one-to-one relationship to anime

**Junction Tables**: `anime_genres`, `anime_studios`, `anime_themes`, `anime_characters` to handle many-to-many relationships.

**Indexing Strategy**: Indexes on frequently queried fields including `mal_id`, `title`, `score`, `rank`, `popularity`, and `year` for optimized search performance.

**Rationale**: This normalized schema design prevents data duplication, maintains referential integrity, and supports complex queries across multiple dimensions (genres, studios, themes, characters). The separation of ML features into a dedicated table allows for independent updates without affecting core anime data.

### Data Ingestion Pipeline

**Pattern**: API-based extraction with rate limiting and error handling.

**Implementation**: 
- Jikan API (unofficial MyAnimeList API) client with 1-second rate limiting between requests
- Batch processing for top anime, genres, characters, and reviews
- SQLAlchemy merge operations for idempotent data ingestion
- Transaction management with rollback on errors

**Rationale**: The Jikan API provides free access to comprehensive MyAnimeList data without requiring official API keys. Rate limiting prevents API throttling, while merge operations allow re-running ingestion without creating duplicates.

### Machine Learning Integration

**Pattern**: Async inference using Hugging Face Inference API.

**Models Used**:
- `facebook/bart-large-mnli` - Zero-shot text classification for synopsis categorization
- Sentiment analysis model for review sentiment scoring

**Implementation**: ML features are processed separately from core data ingestion and stored in a dedicated table, allowing for model updates and reprocessing without affecting base data.

**Rationale**: Hugging Face Inference API eliminates the need for local model hosting and GPU infrastructure. Zero-shot classification provides flexible categorization without training custom models. Separating ML processing allows for iterative improvement and experimentation.

### Session Management

**Pattern**: SQLAlchemy scoped sessions with connection pooling.

**Configuration**:
- Pool size: 5 connections
- Max overflow: 10 connections
- Pool pre-ping enabled for connection health checking
- Streamlit resource caching for engine and session factory

**Rationale**: Connection pooling reduces overhead from repeatedly creating database connections. Scoped sessions ensure thread-safety in the Streamlit multi-user environment. Pre-ping validation prevents stale connection errors.

### State Management

**Pattern**: Streamlit session state for application lifecycle tracking.

**Key States**:
- `db_initialized` - Tracks database table creation
- `data_ingested` - Tracks completion of data ingestion

**Rationale**: Session state prevents redundant initialization operations and provides user feedback on application readiness across page reruns.

## External Dependencies

### Third-Party APIs
- **Jikan API v4** (`https://api.jikan.moe/v4`) - Unofficial MyAnimeList API for anime data, genres, characters, and reviews. No authentication required but rate-limited to 1 request per second.

### Machine Learning Services
- **Hugging Face Inference API** - Serverless inference for text classification and sentiment analysis
  - Optional authentication via `HUGGINGFACE_TOKEN` or `HF_TOKEN` environment variable
  - Falls back to unauthenticated access with rate limits

### Database
- **PostgreSQL** - Primary relational database
  - Connection via `DATABASE_URL` environment variable (required)
  - Expects standard PostgreSQL connection string format

### Python Libraries
- **SQLAlchemy** - ORM and database toolkit
- **Streamlit** - Web application framework
- **Plotly** - Interactive data visualization
- **Pandas** - Data manipulation and analysis
- **Requests** - HTTP client for Jikan API
- **huggingface_hub** - Hugging Face API client

### Environment Variables
- `DATABASE_URL` (required) - PostgreSQL connection string
- `HUGGINGFACE_TOKEN` or `HF_TOKEN` (optional) - Hugging Face API authentication for higher rate limits