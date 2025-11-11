import os
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, scoped_session
from models import Base
import streamlit as st

def get_database_url():
    return os.environ.get('DATABASE_URL')

@st.cache_resource
def get_engine():
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(
        database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )
    return engine

@st.cache_resource
def get_session_factory():
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)
    return Session

def get_session():
    Session = get_session_factory()
    return Session()

def init_database():
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("Database tables created successfully!")

def check_tables_exist():
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    return len(tables) > 0

def get_table_info():
    engine = get_engine()
    inspector = inspect(engine)
    tables_info = {}
    
    for table_name in inspector.get_table_names():
        columns = inspector.get_columns(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        indexes = inspector.get_indexes(table_name)
        
        tables_info[table_name] = {
            'columns': columns,
            'foreign_keys': foreign_keys,
            'indexes': indexes
        }
    
    return tables_info

def get_table_counts():
    from sqlalchemy import text
    session = get_session()
    
    try:
        inspector = inspect(get_engine())
        table_names = inspector.get_table_names()
        
        counts = {}
        for table_name in table_names:
            result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            counts[table_name] = count
        
        return counts
    finally:
        session.close()

def drop_all_tables():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    print("All tables dropped!")

def reset_database():
    drop_all_tables()
    init_database()
    print("Database reset complete!")
