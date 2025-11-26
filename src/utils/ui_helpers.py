"""
Utility functions for the anime database application
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Any, Optional


def create_metrics_row(metrics: Dict[str, Any], columns: int = 4) -> None:
    """Create a row of metrics in Streamlit columns"""
    cols = st.columns(columns)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % columns]:
            if isinstance(value, dict):
                st.metric(label, value.get('value', 'N/A'), value.get('delta'))
            else:
                st.metric(label, value)


def create_bar_chart(
    data: pd.DataFrame, 
    x: str, 
    y: str, 
    title: str,
    orientation: str = 'v',
    color_col: Optional[str] = None,
    height: int = 400
) -> go.Figure:
    """Create a standardized bar chart"""
    fig = px.bar(
        data, 
        x=x, 
        y=y,
        title=title,
        orientation=orientation,
        color=color_col or y,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(height=height)
    return fig


def create_pie_chart(
    data: pd.DataFrame, 
    values: str, 
    names: str, 
    title: str,
    height: int = 400
) -> go.Figure:
    """Create a standardized pie chart"""
    fig = px.pie(
        data, 
        values=values, 
        names=names,
        title=title
    )
    fig.update_layout(height=height)
    return fig


def create_histogram(
    data: pd.DataFrame, 
    x: str, 
    title: str,
    nbins: int = 20,
    height: int = 400
) -> go.Figure:
    """Create a standardized histogram"""
    fig = px.histogram(
        data, 
        x=x, 
        nbins=nbins,
        title=title,
        color_discrete_sequence=["#ff6b6b"]
    )
    fig.update_layout(height=height)
    return fig


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}" if isinstance(num, int) else str(num)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is 0"""
    return numerator / denominator if denominator != 0 else default


def create_dataframe_display(
    data: List[Dict], 
    columns: List[str],
    sort_by: Optional[str] = None,
    ascending: bool = True
) -> pd.DataFrame:
    """Create and format a DataFrame for display"""
    if not data:
        return pd.DataFrame()
    
    df = pd.DataFrame(data, columns=columns)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    return df


def show_error_message(error: Exception, context: str = "") -> None:
    """Display a standardized error message"""
    error_msg = f"Error {context}: {str(error)}" if context else f"Error: {str(error)}"
    st.error(error_msg)


def show_success_message(message: str) -> None:
    """Display a standardized success message"""
    st.success(message)


def show_info_message(message: str) -> None:
    """Display a standardized info message"""
    st.info(message)


def show_warning_message(message: str) -> None:
    """Display a standardized warning message"""
    st.warning(message)


def create_expandable_section(title: str, content: Any, expanded: bool = False) -> None:
    """Create an expandable section with content"""
    with st.expander(title, expanded=expanded):
        if callable(content):
            content()
        else:
            st.write(content)


def validate_session_state_keys(required_keys: List[str]) -> bool:
    """Validate that required session state keys exist"""
    missing_keys = [key for key in required_keys if key not in st.session_state]
    if missing_keys:
        for key in missing_keys:
            st.session_state[key] = False
        return False
    return True


def init_session_state() -> None:
    """Initialize common session state variables"""
    default_states = {
        'db_initialized': False,
        'data_ingested': False,
        'current_page': 'Database Overview'
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value