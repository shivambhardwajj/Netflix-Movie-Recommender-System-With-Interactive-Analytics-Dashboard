import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Netflix Content Recommender System by Shivam Bhardwaj",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme Custom CSS
st.markdown("""
<style>
    /* Global background and text */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }

    /* Main header */
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    /* Recommendation card */
    .recommendation-card {
        background-color: #1c1c1c;
        color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E50914;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.4);
    }

    /* Sidebar and selectbox */
    .stSelectbox > div > div > select {
        background-color: #2c2c2c;
        color: #ffffff;
    }

    .st-bw {
        background-color: #1c1c1c !important;
        color: white !important;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1c1c1c;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load Netflix dataset"""
    try:
        # Try to load local file first
        if os.path.exists('netflix_titles.csv'):
            df = pd.read_csv('netflix_titles.csv')
        else:
            # Download from Kaggle using kagglehub
            import kagglehub
            path = kagglehub.dataset_download("shivamb/netflix-shows")
            print("Path to dataset files:", path)
            df = pd.read_csv(os.path.join(path, 'netflix_titles.csv'))

        # Data preprocessing
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
        df['month_added'] = df['date_added'].dt.month

        # Fill missing values
        df['country'].fillna('Unknown', inplace=True)
        df['cast'].fillna('Unknown', inplace=True)
        df['director'].fillna('Unknown', inplace=True)
        df['rating'].fillna('Not Rated', inplace=True)
        df['listed_in'].fillna('Unknown', inplace=True)
        df['description'].fillna('No description available', inplace=True)

        # Create content features for recommendation
        df['content'] = df['listed_in'] + ' ' + df['description'] + ' ' + df['cast'] + ' ' + df['director']

        return df

    except FileNotFoundError:
        st.error("Please download 'netflix_titles.csv' from Kaggle and place it in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


class NetflixRecommender:
    def __init__(self, df):
        self.df = df
        self.tfidf_matrix = None
        self.content_similarity = None
        self.setup_content_based()

    def setup_content_based(self):
        """Setup content-based recommendation system"""
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = tfidf.fit_transform(self.df['content'].fillna(''))

        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(self.tfidf_matrix)

    def get_content_recommendations(self, title, num_recommendations=10):
        """Get content-based recommendations"""
        try:
            # Find the index of the movie/show
            idx = self.df[self.df['title'].str.contains(title, case=False, na=False)].index[0]

            # Get similarity scores
            sim_scores = list(enumerate(self.content_similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # Get top recommendations (excluding the input title)
            sim_scores = sim_scores[1:num_recommendations + 1]
            movie_indices = [i[0] for i in sim_scores]

            recommendations = self.df.iloc[movie_indices][
                ['title', 'type', 'listed_in', 'release_year', 'rating', 'description']]
            recommendations['similarity_score'] = [score[1] for score in sim_scores]

            return recommendations

        except IndexError:
            return pd.DataFrame()

    def get_popular_recommendations(self, content_type=None, genre=None, num_recommendations=10):
        """Get popular content recommendations"""
        filtered_df = self.df.copy()

        if content_type and content_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == content_type]

        if genre and genre != 'All':
            filtered_df = filtered_df[filtered_df['listed_in'].str.contains(genre, case=False, na=False)]

        # Sort by release year (assuming newer content is more popular)
        popular = filtered_df.sort_values('release_year', ascending=False).head(num_recommendations)

        return popular[['title', 'type', 'listed_in', 'release_year', 'rating', 'description']]


def main():
    # Header
    st.markdown('<div class="main-header">🎬 Netflix Content Recommender</div>', unsafe_allow_html=True)

    # Load data
    df = load_data()
    if df is None:
        st.stop()

    # Initialize recommender
    recommender = NetflixRecommender(df)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page",
                                ["🏠 Dashboard", "🔍 Content Recommendations", "📊 Data Analytics", "🎯 Popular Content"])

    if page == "🏠 Dashboard":
        dashboard_page(df)
    elif page == "🔍 Content Recommendations":
        recommendation_page(df, recommender)
    elif page == "📊 Data Analytics":
        analytics_page(df)
    elif page == "🎯 Popular Content":
        popular_content_page(df, recommender)


def dashboard_page(df):
    st.header("📊 Netflix Content Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Titles", f"{len(df):,}")

    with col2:
        st.metric("Movies", f"{len(df[df['type'] == 'Movie']):,}")

    with col3:
        st.metric("TV Shows", f"{len(df[df['type'] == 'TV Show']):,}")

    with col4:
        latest_year = df['release_year'].max()
        st.metric("Latest Release", str(latest_year))

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Content type distribution
        type_counts = df['type'].value_counts()
        fig_pie = px.pie(values=type_counts.values, names=type_counts.index,
                         title="Content Type Distribution",
                         color_discrete_sequence=['#E50914', '#221F1F'])
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Top countries
        country_counts = df['country'].str.split(',').explode().str.strip().value_counts().head(10)
        fig_bar = px.bar(x=country_counts.values, y=country_counts.index,
                         orientation='h', title="Top 10 Countries by Content",
                         color_discrete_sequence=['#E50914'])
        fig_bar.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Content release timeline
    yearly_counts = df.groupby(['release_year', 'type']).size().reset_index(name='count')
    fig_timeline = px.line(yearly_counts, x='release_year', y='count', color='type',
                           title="Content Release Timeline",
                           color_discrete_sequence=['#E50914', '#221F1F'])
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)


def recommendation_page(df, recommender):
    st.header("🔍 Get Personalized Recommendations")

    # User input
    col1, col2 = st.columns([3, 1])

    with col1:
        selected_title = st.selectbox("Select a title you like:",
                                      options=df['title'].unique(),
                                      index=0)

    with col2:
        num_recs = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Finding similar content..."):
            recommendations = recommender.get_content_recommendations(selected_title, num_recs)

            if not recommendations.empty:
                st.success(f"Found {len(recommendations)} recommendations based on '{selected_title}'")

                # Display recommendations
                for idx, row in recommendations.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>🎬 {row['title']}</h4>
                            <p><strong>Type:</strong> {row['type']} | <strong>Year:</strong> {row['release_year']} | <strong>Rating:</strong> {row['rating']}</p>
                            <p><strong>Genres:</strong> {row['listed_in']}</p>
                            <p><strong>Description:</strong> {row['description'][:200]}...</p>
                            <p><em>Similarity Score: {row['similarity_score']:.3f}</em></p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try selecting a different title.")


def analytics_page(df):
    st.header("📊 Content Analytics Dashboard")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        type_filter = st.selectbox("Content Type", ['All'] + list(df['type'].unique()))

    with col2:
        years = sorted(df['release_year'].dropna().unique(), reverse=True)
        year_filter = st.selectbox("Release Year", ['All'] + [str(int(year)) for year in years])

    with col3:
        ratings = sorted(df['rating'].unique())
        rating_filter = st.selectbox("Rating", ['All'] + list(ratings))

    # Apply filters
    filtered_df = df.copy()
    if type_filter != 'All':
        filtered_df = filtered_df[filtered_df['type'] == type_filter]
    if year_filter != 'All':
        filtered_df = filtered_df[filtered_df['release_year'] == int(year_filter)]
    if rating_filter != 'All':
        filtered_df = filtered_df[filtered_df['rating'] == rating_filter]

    # Genre analysis
    st.subheader("🎭 Genre Analysis")
    genres = filtered_df['listed_in'].str.split(',').explode().str.strip().value_counts().head(15)

    fig_genres = px.bar(x=genres.values, y=genres.index, orientation='h',
                        title="Top Genres", color_discrete_sequence=['#E50914'])
    fig_genres.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_genres, use_container_width=True)

    # Rating distribution
    col1, col2 = st.columns(2)

    with col1:
        rating_counts = filtered_df['rating'].value_counts()
        fig_rating = px.pie(values=rating_counts.values, names=rating_counts.index,
                            title="Rating Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_rating, use_container_width=True)

    with col2:
        # Duration analysis for movies
        if 'Movie' in filtered_df['type'].values:
            movie_durations = filtered_df[filtered_df['type'] == 'Movie']['duration'].str.extract(r'(\d+)').astype(
                float).dropna()
            if not movie_durations.empty:
                fig_duration = px.histogram(movie_durations, x=0, title="Movie Duration Distribution (minutes)",
                                            color_discrete_sequence=['#E50914'])
                fig_duration.update_xaxes(title="Duration (minutes)")
                fig_duration.update_yaxes(title="Count")
                st.plotly_chart(fig_duration, use_container_width=True)

    # Word cloud
    st.subheader("📝 Content Description Word Cloud")
    if st.button("Generate Word Cloud"):
        text = ' '.join(filtered_df['description'].dropna().astype(str))

        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Reds', max_words=100).generate(text)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


def popular_content_page(df, recommender):
    st.header("🎯 Popular Content")

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        content_type = st.selectbox("Content Type", ['All', 'Movie', 'TV Show'])

    with col2:
        genres = ['All'] + list(
            set([genre.strip() for sublist in df['listed_in'].str.split(',').dropna() for genre in sublist]))
        selected_genre = st.selectbox("Genre", genres)

    with col3:
        num_results = st.slider("Number of results", 5, 50, 20)

    # Get popular recommendations
    popular_content = recommender.get_popular_recommendations(content_type, selected_genre, num_results)

    st.subheader(f"📈 Top {len(popular_content)} Popular Content")

    # Display in a nice table
    for idx, row in popular_content.iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown(f"""
                **{row['title']}**
                - Type: {row['type']}
                - Year: {row['release_year']}
                - Rating: {row['rating']}
                """)

            with col2:
                st.markdown(f"""
                **Genres:** {row['listed_in']}

                **Description:** {row['description'][:300]}...
                """)

            st.markdown("---")


if __name__ == "__main__":
    main()

st.markdown("""
<style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #888;
        text-align: center;
        font-size: 0.9rem;
        padding: 10px 0;
    }
</style>
<div class="footer">
    Built by <strong>Shivam Bhardwaj</strong>
</div>
""", unsafe_allow_html=True)