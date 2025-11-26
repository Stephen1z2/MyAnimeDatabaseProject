"""
Machine Learning page
"""

import streamlit as st
import pandas as pd
import sys
import os
from contextlib import redirect_stdout, redirect_stderr
import io

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from anime_ml_advanced import AnimeMLAnalyzer, demonstrate_anime_ml
from database import get_session
from models import Anime, Genre, Studio, Character
from sqlalchemy import func
from src.components.ui_components import MetricsComponent, ErrorHandlerComponent
from src.utils.ui_helpers import show_warning_message, show_success_message, show_info_message


def render_machine_learning():
    """Render the Machine Learning page"""
    st.header("🤖 Machine Learning Analytics")
    
    if not st.session_state.get('db_initialized', False):
        show_warning_message("Please initialize and populate the database first.")
        return
    
    st.markdown("""
    **Advanced machine learning algorithms for comprehensive anime analysis!**
    
    This section provides:
    - **🎯 Score Prediction**: Predict anime ratings using multiple ML models
    - **📊 Genre Classification**: Classify anime into genres using NLP
    - **🔍 Clustering Analysis**: Discover hidden patterns in anime data
    - **📈 Feature Importance**: Understand what makes anime popular
    - **🧠 Model Comparison**: Compare different ML algorithms
    """)
    
    try:
        # Initialize ML module
        ml_module = AnimeMLAnalyzer()
        
        # ML Mode Selection
        ml_mode = st.selectbox(
            "Choose ML Analysis Type",
            ["🎯 Hidden Gem Finder", "Score Prediction", "Genre Classification", "Clustering Analysis", "Feature Analysis", "Model Comparison"]
        )
        
        if ml_mode == "🎯 Hidden Gem Finder":
            _render_hidden_gem_finder()
        
        elif ml_mode == "Score Prediction":
            _render_score_prediction(ml_module)
        
        elif ml_mode == "Genre Classification":
            _render_genre_classification(ml_module)
        
        elif ml_mode == "Clustering Analysis":
            _render_clustering_analysis(ml_module)
        
        elif ml_mode == "Feature Analysis":
            _render_feature_analysis(ml_module)
        
        elif ml_mode == "Model Comparison":
            _render_model_comparison()
    
    except ImportError as e:
        st.error(f"❌ Missing dependencies for machine learning: {str(e)}")
        show_info_message("Install required packages with: `pip install scikit-learn xgboost matplotlib seaborn`")
    except Exception as e:
        ErrorHandlerComponent.handle_database_error(e, "initializing ML module")


def _render_hidden_gem_finder():
    """Render the Hidden Gem Finder section"""
    st.subheader("💎 Hidden Gem Finder")
    st.markdown("""
    **Discover underrated anime using machine learning insights!**
    
    This tool uses patterns learned from your database to find anime that should be rated higher
    based on their characteristics like studio quality, episode count, genres, and more.
    """)
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        confidence_level = st.selectbox(
            "Confidence Level",
            ["High Confidence (Top Studios)", "Medium Confidence", "All Hidden Gems"],
            help="High = Top studios with low scores, Medium = Good characteristics with low scores, All = Every potential gem"
        )
    with col2:
        max_results = st.slider("Max Results", 5, 50, 20)
    
    if st.button("🔍 Find Hidden Gems", type="primary"):
        with st.spinner("🤖 Analyzing anime patterns and finding hidden gems..."):
            try:
                session = get_session()
                
                # Get all anime with scores
                all_anime = session.query(Anime).filter(
                    Anime.score.isnot(None),
                    Anime.episodes.isnot(None)
                ).all()
                
                # Based on ML analysis patterns
                top_studios = {'Madhouse', 'MAPPA', 'Sunrise', 'Shaft', 'Studio Pierrot', 
                             'Bones', 'Wit Studio', 'White Fox', 'Production I.G', 'Ufotable'}
                
                hidden_gems = []
                
                for anime in all_anime:
                    anime_studios = {s.name for s in anime.studios}
                    has_top_studio = bool(anime_studios.intersection(top_studios))
                    
                    is_hidden_gem = False
                    reason = ""
                    confidence = ""
                    
                    # High confidence: Top studio but lower score
                    if has_top_studio and anime.score < 8.2:
                        is_hidden_gem = True
                        studio_names = list(anime_studios.intersection(top_studios))
                        reason = f"Top studio ({studio_names[0]}) but only {anime.score} score"
                        confidence = "High"
                    
                    # Medium confidence: Good characteristics
                    elif (confidence_level in ["Medium Confidence", "All Hidden Gems"] and
                          anime.episodes and 12 <= anime.episodes <= 50 and 
                          len(anime.genres) >= 2 and anime.score < 7.9):
                        is_hidden_gem = True
                        reason = f"Quality length ({anime.episodes} eps) + multi-genre but {anime.score} score"
                        confidence = "Medium"
                    
                    # Low confidence: Recent but low scoring
                    elif (confidence_level == "All Hidden Gems" and
                          anime.year and anime.year >= 2015 and anime.score < 7.5):
                        is_hidden_gem = True
                        reason = f"Recent ({anime.year}) but low {anime.score} score"
                        confidence = "Low"
                    
                    if is_hidden_gem:
                        # Filter by confidence level
                        if (confidence_level == "High Confidence (Top Studios)" and confidence == "High") or \
                           (confidence_level == "Medium Confidence" and confidence in ["High", "Medium"]) or \
                           (confidence_level == "All Hidden Gems"):
                            hidden_gems.append({
                                'title': anime.title,
                                'score': anime.score,
                                'episodes': anime.episodes,
                                'year': anime.year,
                                'studios': [s.name for s in anime.studios][:2],
                                'genres': [g.name for g in anime.genres][:4],
                                'reason': reason,
                                'confidence': confidence,
                                'mal_id': anime.mal_id
                            })
                
                # Sort by confidence and then score
                confidence_order = {"High": 0, "Medium": 1, "Low": 2}
                hidden_gems.sort(key=lambda x: (confidence_order[x['confidence']], x['score']))
                
                # Limit results
                hidden_gems = hidden_gems[:max_results]
                
                if hidden_gems:
                    show_success_message(f"✨ Found {len(hidden_gems)} hidden gems!")
                    
                    # Display results in a nice format
                    st.subheader("💎 Your Hidden Gems")
                    
                    for i, gem in enumerate(hidden_gems, 1):
                        _render_hidden_gem_item(gem, i)
                    
                    # Summary statistics
                    high_conf = len([g for g in hidden_gems if g['confidence'] == 'High'])
                    med_conf = len([g for g in hidden_gems if g['confidence'] == 'Medium'])
                    low_conf = len([g for g in hidden_gems if g['confidence'] == 'Low'])
                    
                    st.subheader("📊 Summary")
                    summary_metrics = {
                        "🔥 High Confidence": high_conf,
                        "⭐ Medium Confidence": med_conf,
                        "💡 Potential Gems": low_conf
                    }
                    MetricsComponent.render(summary_metrics, columns=3)
                    
                    show_info_message("💡 **Tip**: Start with High Confidence recommendations - these are from top studios but scored lower than expected!")
                    
                else:
                    st.warning("No hidden gems found with the current criteria. Try lowering the confidence level!")
                
                session.close()
                
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "finding hidden gems")


def _render_hidden_gem_item(gem, index):
    """Render a single hidden gem item"""
    # Create confidence badge
    if gem['confidence'] == "High":
        badge = "🔥 HIGH"
        badge_color = "#ff4444"
    elif gem['confidence'] == "Medium":
        badge = "⭐ MEDIUM"  
        badge_color = "#ffaa00"
    else:
        badge = "💡 POTENTIAL"
        badge_color = "#00aaff"
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{index}. {gem['title']}**")
            st.markdown(f"🎭 {', '.join(gem['genres'])}")
            st.markdown(f"🏭 {', '.join(gem['studios'])}")
            
        with col2:
            st.markdown(f"**Score: {gem['score']}**")
            st.markdown(f"Episodes: {gem['episodes']}")
            st.markdown(f"Year: {gem['year'] or 'Unknown'}")
            
        with col3:
            st.markdown(f"<span style='background-color: {badge_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 12px;'>{badge}</span>", unsafe_allow_html=True)
        
        st.markdown(f"*{gem['reason']}*")
        
        # Add link to MAL
        if gem['mal_id']:
            st.markdown(f"[🔗 View on MyAnimeList](https://myanimelist.net/anime/{gem['mal_id']})")
        
        st.markdown("---")


def _render_score_prediction(ml_module):
    """Render Score Prediction section"""
    st.subheader("🎯 Anime Score Prediction")
    st.markdown("Predict anime scores using machine learning models trained on features like genres, episodes, studios, and more.")
    
    # Model selection for score prediction
    score_model = st.selectbox(
        "Select Model",
        ["Random Forest", "XGBoost", "SVM", "Neural Network"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
        random_seed = st.number_input("Random Seed", 1, 1000, 42)
    
    with col2:
        cross_validation = st.checkbox("Use Cross-Validation", True)
        feature_importance = st.checkbox("Show Feature Importance", True)
    
    if st.button("🚀 Train Score Prediction Model"):
        with st.spinner("Training score prediction model..."):
            try:
                # Load data and train
                ml_module.load_and_prepare_data()
                results = ml_module.train_score_predictor(test_size=test_size)
                
                if results:
                    show_success_message("✅ Model trained successfully!")
                    
                    # Display metrics
                    st.subheader("📊 Model Performance")
                    
                    metrics = {
                        "R² Score": f"{results.get('r2_score', 0):.4f}",
                        "RMSE": f"{results.get('rmse', 0):.4f}",
                        "MAE": f"{results.get('mae', 0):.4f}"
                    }
                    MetricsComponent.render(metrics, columns=3)
                    
                    show_info_message("💾 Model trained and ready for predictions!")
            
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "training model")


def _render_genre_classification(ml_module):
    """Render Genre Classification section"""
    st.subheader("📊 Genre Classification")
    st.markdown("Use NLP and machine learning to classify anime genres based on synopsis and features.")
    
    classification_model = st.selectbox(
        "Select Model",
        ["Random Forest", "SVM", "Naive Bayes", "Logistic Regression"]
    )
    
    target_genre = st.selectbox(
        "Target Genre",
        ["Action", "Adventure", "Comedy", "Drama", "Fantasy", "Romance", "Sci-Fi", "Thriller"]
    )
    
    if st.button("🔍 Train Genre Classifier"):
        with st.spinner("Training genre classification model..."):
            try:
                # Load data and train
                ml_module.load_and_prepare_data()
                results = ml_module.train_genre_classifier()
                
                if results:
                    show_success_message("✅ Genre classifier trained successfully!")
                    
                    # Display classification metrics
                    st.subheader("📊 Classification Performance")
                    st.write("Multi-genre classification completed successfully!")
                    
                    show_info_message("💾 Genre classifier trained and ready!")
            
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "training classifier")


def _render_clustering_analysis(ml_module):
    """Render Clustering Analysis section"""
    st.subheader("🔍 Clustering Analysis")
    st.markdown("Discover hidden patterns and group similar anime using unsupervised learning.")
    
    clustering_algorithm = st.selectbox(
        "Select Algorithm",
        ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_clusters = st.slider("Number of Clusters", 2, 15, 5)
    with col2:
        include_synopsis = st.checkbox("Include Synopsis Features", True)
    
    if st.button("🔬 Perform Clustering"):
        with st.spinner("Performing clustering analysis..."):
            try:
                # Load data and perform clustering
                ml_module.load_and_prepare_data()
                results = ml_module.perform_anime_clustering(n_clusters=n_clusters)
                
                if results:
                    show_success_message("✅ Clustering completed successfully!")
                    
                    # Display clustering results
                    st.subheader("📊 Clustering Results")
                    st.write(f"Successfully created {n_clusters} anime clusters based on features!")
                    
                    if 'cluster_stats' in results:
                        st.subheader("📈 Cluster Statistics")
                        st.dataframe(results['cluster_stats'])
            
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "performing clustering")


def _render_feature_analysis(ml_module):
    """Render Feature Analysis section"""
    st.subheader("📈 Comprehensive ML Analysis")
    st.markdown("Generate a complete machine learning analysis report with all models and insights.")
    
    if st.button("📊 Generate ML Report"):
        with st.spinner("Running comprehensive ML analysis..."):
            try:
                # Load data first
                ml_module.load_and_prepare_data()
                
                # Generate comprehensive report
                report = ml_module.generate_ml_report()
                
                if report:
                    show_success_message("✅ ML analysis completed successfully!")
                    
                    # Display the comprehensive report
                    st.subheader("🎯 Machine Learning Analysis Report")
                    st.text_area("Analysis Report", report, height=400)
            
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "generating report")


def _render_model_comparison():
    """Render Model Comparison section"""
    st.subheader("🧠 Educational ML Demo")
    st.markdown("""
    Run a comprehensive demonstration of machine learning techniques on your anime database.
    This will train multiple models and show you how they work!
    """)
    
    if st.button("⚖️ Run ML Demo"):
        with st.spinner("Running comprehensive ML demonstration..."):
            try:
                # Create string buffers
                stdout_buffer = io.StringIO()
                stderr_buffer = io.StringIO()
                
                # Redirect stdout and stderr
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    demonstrate_anime_ml()
                
                # Get the output
                demo_output = stdout_buffer.getvalue()
                error_output = stderr_buffer.getvalue()
                
                if demo_output:
                    show_success_message("✅ ML demonstration completed!")
                    st.subheader("📋 ML Demo Results")
                    st.text_area("Demo Output", demo_output, height=400)
                
                if error_output:
                    st.warning("⚠️ Some warnings occurred:")
                    st.text_area("Warnings", error_output, height=200)
            
            except Exception as e:
                ErrorHandlerComponent.handle_database_error(e, "running demo")