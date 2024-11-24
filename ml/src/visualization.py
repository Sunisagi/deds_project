import streamlit as st
import numpy as np
import pandas as pd
from component import *
from datetime import datetime, timezone
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from recommendation import TextRecommender,load_latest_file,load_latest_pkl
import ast
# Main content
st.title('Project')

# Cache functions
@st.cache_resource  # 👈 Add the caching decorator
def load_model_recomendation():
    recommender_dir = "/model/recommender"
    return load_latest_pkl("model",recommender_dir)

@st.cache_data  # 👈 Add the caching decorator
def load_file(prefix):
    data_dir = '/process'
    return load_latest_file(prefix,data_dir)


# Session state initialization for mode and paper_id
if 'mode' not in st.session_state:
    st.session_state.mode = "Data  Visualization"
if 'paper_id' not in st.session_state:
    st.session_state.paper_id = None
if 'author_id' not in st.session_state:
    st.session_state.author_id = None
model = load_model_recomendation()
papers_df = load_file("paper")
authors_df = load_file("author")
affiliations_df = load_file("affiliation")

papers_df['id'] = papers_df['id'].astype(str)
affiliations_df['affid'] = affiliations_df['affid'].astype(str)
authors_df['auid'] = authors_df['auid'].astype(str)
authors_df['affliation'] = authors_df['affliation'].astype(str)

st.sidebar.header('Mode')

def on_change_mode():
    st.session_state.mode = st.session_state.sidebar_option
    if st.session_state.mode == "Paper Information":
        st.session_state.author_id = None
    elif st.session_state.mode == "Author Information":
        st.session_state.paper_id = None
    else:
        st.session_state.paper_id = None
        st.session_state.author_id = None

# Create the sidebar selectbox with a callback
option = st.sidebar.selectbox(
    'Choose Mode',
    ['Data  Visualization','Recommendation', 'Paper Information', 'Author Information'],
    index=['Data  Visualization','Recommendation', 'Paper Information', 'Author Information'].index(st.session_state.mode),
    key='sidebar_option',  # Give it a key in session state
    on_change=on_change_mode  # Add callback function
)

st.sidebar.write('You selected:', option)

if st.session_state.mode :
    st.write(st.session_state.mode)
    st.write(f"Received paper: {st.session_state.paper_id}")
    st.write(f"Received author: {st.session_state.author_id}")
# Show content based on the selected mode
if st.session_state.mode == "Recommendation":
    with st.container():
        st.header('Paper Recommendation')
        st.write('Please fill the Query Text for the paper recommendation')
    create_recommendation_interface(model)

elif st.session_state.mode == "Paper Information":
    with st.container():
        st.header('Paper Information')
        st.write('Please provide the Paper ID to get detailed information')

    # Auto-fill text_input if paper_id is not None
    paper_id = st.text_input("Search For Paper", value=st.session_state.paper_id if st.session_state.paper_id else "")

    # Update paper_id in session_state when user provides a new input
    if paper_id:
        st.session_state.paper_id = paper_id

    # Automatically fetch and display the paper details when paper_id is assigned
    if st.session_state.paper_id:
        paper_id = st.session_state.paper_id  # Use the session paper_id directly
        
        # Check if the paper_id exists in the papers dataframe
        if paper_id in papers_df['id'].values:
            # Find the paper details
            paper_row = papers_df[papers_df['id'] == paper_id].iloc[0]
            st.subheader(f"Paper ID: {paper_row['id']}")
            st.write(f"**Publish Date:** {paper_row['date']}")
            st.write(f"**Title:** {paper_row['title']}")
            st.write(f"**Abstract:** {paper_row['description']}")
            affiliated_ids = paper_row['affiliations']
            affiliations_filtered = affiliations_df[affiliations_df['affid'].isin(affiliated_ids)]

            authors_ids = paper_row['authors']
            authors_filtered = authors_df[authors_df['auid'].isin(authors_ids)]
            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)
            # Display the related affiliations
            if not affiliations_filtered.empty:
                with st.expander("### Affiliations related to this paper:"):
                    st.dataframe(affiliations_filtered[['affid', 'name', 'country', 'city']])
            else:
                st.write("No matching affiliations found.")
            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)
            
            if not authors_filtered.empty:
                with st.expander("### Authors related to this paper:"):
                    create_author(authors_filtered)
            else:
                st.write("No matching authors found.")
            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

            with st.expander("### Similar to this paper:"):
                create_table(model.find_similar_papers(paper_id,10)[1],False)
            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)
        else:
            st.write("No paper found with that ID.")
elif st.session_state.mode == "Author Information":
    with st.container():
        st.header('Author Information')
        st.write('Please provide the Author ID to get detailed information.')

    # Auto-fill text_input if author_id is not None
    author_id = st.text_input("Search For Author", value=st.session_state.author_id if "author_id" in st.session_state else "")

    # Update author_id in session_state when user provides a new input
    if author_id:
        st.session_state.author_id = author_id

    # Automatically fetch and display the author details when author_id is assigned
    if st.session_state.author_id:
        author_id = st.session_state.author_id  # Use the session author_id directly

        # Check if the author_id exists in the authors dataframe
        if author_id in authors_df['auid'].values:
            # Find the author details
            author_row = authors_df[authors_df['auid'] == author_id].iloc[0]
            st.subheader(f"Author ID: {author_row['auid']}")
            st.write(f"**Given Name:** {author_row['given-name']}")
            st.write(f"**Initials:** {author_row['initials']}")
            st.write(f"**Surname:** {author_row['surname']}")
            st.write(f"**Indexed Name:** {author_row['indexed-name']}")
            st.write(author_row['affliation'])
            # Ensure affliation is a list, process each element, and create a list of aff_ids
            au_affliations = ast.literal_eval(author_row['affliation'])
            if isinstance(au_affliations, str):
                # If affliation is a string, convert it to a single-element list
                aff_ids = [str(au_affliations)]
            elif isinstance(au_affliations, list):
                # If affliation is already a list, process each element
                aff_ids = [str(aff) for aff in au_affliations]
            else:
                aff_ids = []  # Handle unexpected cases gracefully

            st.write(aff_ids) 

            # Get the related papers
            related_papers = papers_df[papers_df['authors'].apply(lambda x: author_id in x if isinstance(x, list) else False)]
            
            affiliations_filtered = affiliations_df[affiliations_df['affid'].isin(aff_ids)]

            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

            # Display the related papers
            if not related_papers.empty:
                with st.expander("### Papers written by this author:"):
                     create_normal_table(related_papers)
            else:
                st.write("No related papers found.")

            st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

            # Display the related affiliations
            if not affiliations_filtered.empty:
                with st.expander("### Affiliations associated with this author:"):
                    st.dataframe(affiliations_filtered[['affid', 'name', 'country', 'city']])
            else:
                st.write("No related affiliations found.")
        else:
            st.write("No author found with that ID.")

else:

    st.title("Dataset Visualization")

    # --- First Visualization: Group Papers by Number of Authors ---
    st.subheader("1. Papers Grouped by Number of Authors")

    # Compute number of authors per paper
    papers_df["author_count"] = papers_df["authors"].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Group by author count
    papers_grouped = papers_df.groupby("author_count").size().reset_index(name="count")

    # All groups
    grouped_bar_all = px.bar(
        papers_grouped,
        x="author_count",
        y="count",
        title="All Papers Grouped by Number of Authors",
        labels={"author_count": "Number of Authors", "count": "Number of Papers"},
    )
    st.plotly_chart(grouped_bar_all, use_container_width=True)

    # Groups with more than 2 authors per paper
    papers_grouped_filtered = papers_grouped[papers_grouped["author_count"] > 20]
    grouped_bar_filtered = px.bar(
        papers_grouped_filtered,
        x="author_count",
        y="count",
        title="Papers with More Than 2 Authors Grouped by Number of Authors",
        labels={"author_count": "Number of Authors", "count": "Number of Papers"},
    )
    st.plotly_chart(grouped_bar_filtered, use_container_width=True)

    # --- Second Visualization: Group Authors by Number of Papers ---
    st.subheader("2. Authors Grouped by Number of Papers")

    # Compute number of papers per author
    author_paper_counts = {}
    for authors in papers_df["authors"]:
        if isinstance(authors, list):
            for author in authors:
                author_paper_counts[author] = author_paper_counts.get(author, 0) + 1

    # Convert to DataFrame
    author_paper_df = pd.DataFrame(list(author_paper_counts.items()), columns=["Author ID", "Paper Count"])

    # Group by paper count
    author_grouped = author_paper_df.groupby("Paper Count").size().reset_index(name="count")

    # All groups
    author_bar_all = px.bar(
        author_grouped,
        x="Paper Count",
        y="count",
        title="All Authors Grouped by Number of Papers",
        labels={"Paper Count": "Number of Papers", "count": "Number of Authors"},
    )
    st.plotly_chart(author_bar_all, use_container_width=True)

    # Groups with more than 2 papers per author
    author_grouped_filtered = author_grouped[author_grouped["Paper Count"] > 10]
    author_bar_filtered = px.bar(
        author_grouped_filtered,
        x="Paper Count",
        y="count",
        title="Authors with More Than 2 Papers Grouped by Number of Papers",
        labels={"Paper Count": "Number of Papers", "count": "Number of Authors"},
    )
    st.plotly_chart(author_bar_filtered, use_container_width=True)

    # # --- Third Visualization: Elbow Method ---
    # st.subheader("3. Elbow Method for Clustering")

    # # Generate similarity matrices for titles and abstracts
    # title_matrix = model.title_similarity_matrix  # Replace with precomputed matrix
    # abstract_matrix = model.abstract_similarity_matrix # Replace with precomputed matrix

    # def elbow_method(matrix, max_clusters=10):
    #     distortions = []
    #     for k in range(2, max_clusters + 1):
    #         kmeans = KMeans(n_clusters=k, random_state=42)
    #         kmeans.fit(matrix)
    #         distortions.append(kmeans.inertia_)
    #     return distortions

    # # Elbow for title
    # title_distortions = elbow_method(title_matrix)
    # abstract_distortions = elbow_method(abstract_matrix)

    # # Create Elbow Chart
    # elbow_chart = go.Figure()
    # elbow_chart.add_trace(go.Scatter(x=list(range(2, 11)), y=title_distortions, mode='lines+markers', name="Title"))
    # elbow_chart.add_trace(go.Scatter(x=list(range(2, 11)), y=abstract_distortions, mode='lines+markers', name="Abstract"))
    # elbow_chart.update_layout(
    #     title="Elbow Method for Clustering",
    #     xaxis_title="Number of Clusters",
    #     yaxis_title="Distortion",
    # )
    # st.plotly_chart(elbow_chart, use_container_width=True)

    # # --- Fourth Visualization: Clustering ---
    # st.subheader("4. Clustering Visualization")

    # def plot_clusters(matrix, n_clusters, title):
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     labels = kmeans.fit_predict(matrix)
        
    #     # Reduce dimensions for visualization
    #     pca = PCA(n_components=2)
    #     reduced_data = pca.fit_transform(matrix)
        
    #     # Create Scatter Plot
    #     cluster_chart = px.scatter(
    #         x=reduced_data[:, 0],
    #         y=reduced_data[:, 1],
    #         color=labels.astype(str),
    #         title=title,
    #         labels={"x": "PCA 1", "y": "PCA 2", "color": "Cluster"},
    #     )
    #     return cluster_chart

    # # Plot Clusters
    # optimal_title_clusters = 4  # Replace with the result of elbow method analysis for title
    # optimal_abstract_clusters = 4  # Replace with the result of elbow method analysis for abstract

    # st.write("### Clusters based on Title Similarity")
    # st.plotly_chart(plot_clusters(title_matrix, optimal_title_clusters, "Clusters (Title Similarity)"), use_container_width=True)

    # st.write("### Clusters based on Abstract Similarity")
    # st.plotly_chart(plot_clusters(abstract_matrix, optimal_abstract_clusters, "Clusters (Abstract Similarity)"), use_container_width=True)