# Running MyAnimeList Database Project Locally

This guide will help you run the MyAnimeList Database Project on your local computer.

## Prerequisites

Before you begin, make sure you have the following installed on your computer:

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - Verify installation: `python --version` or `python3 --version`

2. **PostgreSQL Database**
   - Download from: https://www.postgresql.org/download/
   - Or use Docker: `docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=yourpassword postgres`

3. **Git** (to clone the repository)
   - Download from: https://git-scm.com/downloads/

## Step 1: Get the Code

### Option A: Clone from GitHub (after you push it)
```bash
git clone https://github.com/Stephen1z2/MyNewDatabaseAttempt.git
cd MyNewDatabaseAttempt
```

## Step 2: Set Up Python Virtual Environment (Recommended)

This keeps your project dependencies isolated from other Python projects.

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

## Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt.local
```

Or install packages individually:
```bash
pip install streamlit sqlalchemy psycopg2-binary requests huggingface-hub pandas plotly python-dotenv
```

## Step 4: Set Up PostgreSQL Database

### Create a Database

1. Open PostgreSQL command line or use pgAdmin
2. Create a new database:
```sql
CREATE DATABASE anime_database;
```

3. Note your database connection details:
   - Host: `localhost` (usually)
   - Port: `5432` (default)
   - Username: Your PostgreSQL username (often `postgres`)
   - Password: Your PostgreSQL password
   - Database: `anime_database`

## Step 5: Configure Environment Variables

Create a `.env` file in your project root directory:

```bash
# Required: Database connection
DATABASE_URL=postgresql://username:password@localhost:5432/anime_database

# Optional: Hugging Face token for better rate limits
HUGGINGFACE_TOKEN=your_token_here
```

**Replace**:
- `username` with your PostgreSQL username
- `password` with your PostgreSQL password
- `your_token_here` with your Hugging Face token (optional)

### Getting a Hugging Face Token (Optional)
1. Go to https://huggingface.co/
2. Create a free account
3. Go to Settings → Access Tokens
4. Create a new token
5. Copy and paste it into your `.env` file

## Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Step 7: Initialize and Populate the Database

1. Go to the "Database Overview" page
2. Click "Initialize Database" to create all tables
3. Go to "Data Ingestion" page
4. Click "Run Full Ingestion" to populate with real anime data

## Troubleshooting

### Database Connection Error
**Problem**: `could not connect to server`

**Solution**:
- Make sure PostgreSQL is running
- Check your DATABASE_URL in `.env` file
- Verify username, password, and database name are correct

### Module Not Found Error
**Problem**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
- Make sure your virtual environment is activated
- Run `pip install -r requirements.txt` again

### Port Already in Use
**Problem**: `Address already in use`

**Solution**:
- Run Streamlit on a different port: `streamlit run app.py --server.port 8502`

### Hugging Face Rate Limit
**Problem**: Too many requests to Hugging Face API

**Solution**:
- Add your HUGGINGFACE_TOKEN to the `.env` file
- The app works without a token but has lower rate limits

## Project Structure

```
MyNewDatabaseAttempt/
├── app.py                    # Main Streamlit application
├── models.py                 # Database models (SQLAlchemy)
├── database.py               # Database connection utilities
├── jikan_ingestion.py        # Jikan API data ingestion
├── ml_features.py            # Hugging Face ML integration
├── requirements.txt.local    # Python dependencies for local setup
├── pyproject.toml           # Project configuration
├── .env                     # Environment variables (create this)
├── README.md                # Project documentation
├── GIT_SETUP.md             # GitHub setup guide
└── LOCAL_SETUP.md           # This file - local setup guide
```

## Stopping the Application

- Press `Ctrl+C` in the terminal where Streamlit is running
- To deactivate the virtual environment: `deactivate`

## Updating the Code

If you make changes and want to see them:
1. Save your files
2. Streamlit will automatically reload (you'll see a "Rerun" button)
3. Or stop and restart: `Ctrl+C` then `streamlit run app.py`

## Next Steps

- Explore the database schema in the "Database Schema" page
- Search for anime in the "Search" page
- View analytics in the "Analytics" page
- Check out the code to understand how it works!

## Need Help?

- Check the main README.md for project details
- PostgreSQL documentation: https://www.postgresql.org/docs/
- Streamlit documentation: https://docs.streamlit.io/
- SQLAlchemy documentation: https://docs.sqlalchemy.org/
