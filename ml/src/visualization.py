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
import torch
import ast
import networkx as nx
import pydeck as pdk
from pyvis.network import Network
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
    df = load_latest_file(prefix,data_dir)
    df = df.fillna('N/A')
    return df
# @st.cache_data
# def load_network():
#     return pd.DataFrame(model.find_similar_papers_all(N=50)).reset_index()

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
# papers_network_df = load_network()


papers_df['id'] = papers_df['id'].astype(str)
affiliations_df['affid'] = affiliations_df['affid'].astype(str)
authors_df['auid'] = authors_df['auid'].astype(str)
authors_df['affiliation'] = authors_df['affiliation'].astype(str)

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

# if st.session_state.mode :
#     st.write(st.session_state.mode)
#     st.write(f"Received paper: {st.session_state.paper_id}")
#     st.write(f"Received author: {st.session_state.author_id}")
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

            with st.expander("### Paper Network"):
                #=============================================================================

                # Function to build graph starting from a node
                def build_graph_from_node(start_node_id, connection_range=10, max_depth=2):
                    G = nx.Graph()
                    visited = set()  # To track visited nodes
                    node_depth = {}  # To store the depth of each node
                    # Start with the given node
                    to_visit = [(start_node_id, 0)]  # (node_id, depth)
                    visited.add(start_node_id)
                    node_depth[start_node_id] = 0  # Set the depth of the start node
                    
                    while to_visit:
                        current_node, current_depth = to_visit.pop(0)
                        
                        if current_depth < max_depth:
                            # Get the similar papers for the current node using the model's find_similar_papers method
                            similar_papers_df = model.find_similar_papers(current_node, connection_range*(current_depth+1))[1]  # Get the DataFrame
                            # Sort by score (high to low) and limit to the top N (connection_range)
                            similar_papers_df = similar_papers_df.sort_values(by="similarity_score", ascending=False)
                            
                            # Use connection_range to limit the number of similar papers considered
                            similar_papers = similar_papers_df['id'].tolist()[:connection_range*(current_depth+1)]
                            
                            for similar_paper in similar_papers:
                                if similar_paper != current_node and similar_paper not in visited:
                                    visited.add(similar_paper)
                                    G.add_edge(current_node, similar_paper)
                                    node_depth[similar_paper] = current_depth + 1  # Set the depth of the similar paper
                                    to_visit.append((similar_paper, current_depth + 1))  # Add to the visit queue for the next level

                    return G, node_depth

                # Function to assign colors based on depth
                def assign_colors_by_depth(node_depth):
                    colors = {
                        0: 'red',      # Parent node (level 0)
                        1: 'green',    # Child node (level 1)
                        2: 'blue',     # Grandchild node (level 2)
                        3: 'orange', 
                    }
                    node_colors = [colors[node_depth[node]] if node_depth[node] in colors else 'gray' for node in node_depth]
                    return node_colors
                # Build the graph from the starting node (max_depth = 2, you can increase this if needed)
                max_depth = 3  # Define max depth for traversal
                connection_range = st.slider("Select the number of similar papers (Connections)", min_value=5, max_value=25, value=10, step=1)
                G, node_depth = build_graph_from_node(paper_id, connection_range=connection_range, max_depth=max_depth)
                # Create Pyvis Network object
                net = Network(height='800px', width='100%', notebook=True)

                for node in G.nodes:
                    # Assign color based on depth
                    color = assign_colors_by_depth(node_depth)[list(node_depth.keys()).index(node)]
                    net.add_node(node, color=color, title=str(node))  # Use color based on depth

                for edge in G.edges:
                    net.add_edge(edge[0], edge[1], color="rgba(0,0,0,0)", width=0)

                # Set options for the visualization (this allows better customization of how the graph is displayed)
                net.set_options("""
                var options = {
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.1,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                    }
                },
                "nodes": {
                    "size": 20,
                    "font": {
                    "size": 25,
                    "face": "arial",
                    "weight": "bold"
                    }
                },
                "edges": {
                    "width": 4,
                    "color": {
                    "highlight": "#FF0000"
                    },
                    "smooth": {
                    "type": "continuous"
                    }
                }
                }
                """)


                # Display the network in Streamlit
                st.write("Network Graph:")
                net.show("graph.html")


                import streamlit.components.v1 as st_components
                st_components.html(open("graph.html", "r").read(), height=800)

                #=============================================================================


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
            # st.write(author_row['affiliation'])
            # Ensure affiliation is a list, process each element, and create a list of aff_ids
            if author_row['affiliation'] != "N/A":
                au_affiliations = ast.literal_eval(author_row['affiliation'])
                if isinstance(au_affiliations, str):
                    # If affiliation is a string, convert it to a single-element list
                    aff_ids = [str(au_affiliations)]
                elif isinstance(au_affiliations, list):
                    # If affiliation is already a list, process each element
                    aff_ids = [str(aff) for aff in au_affiliations]
                else:
                    aff_ids = []  # Handle unexpected cases gracefully
            else:
                aff_ids = []

            # st.write(aff_ids) 

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
    st.subheader("1. Number of Papers per Year")

    num_paper_per_year = papers_df.groupby("year").size().reset_index(name="count")

    # All groups

    grouped_fig = go.Figure()
    grouped_fig.add_trace(go.Scatter(
        x=num_paper_per_year["year"],
        y=num_paper_per_year["count"],
        mode='lines+markers',
        name='Line',
    ))

    grouped_fig.update_layout(
        yaxis=dict(range=[0, num_paper_per_year["count"].max()*1.2]),
        title="Number of Papers per Year",
        xaxis_title="Year",
        yaxis_title="Number of Papers"
        )
    st.plotly_chart(grouped_fig, use_container_width=True)

    # Groups with more than 2 authors per paper

    num_paper_cat_year = papers_df.explode('class').groupby(['class', 'year']).size().reset_index(name='count')

    category_list = num_paper_cat_year['class'].unique().tolist()
    year_list = num_paper_cat_year['year'].unique().tolist()

    top_3_cat = num_paper_cat_year.groupby('class').sum().nlargest(3, 'count')

    st.subheader("Top 3 Category with the Most number of Papers")

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        value = top_3_cat.iloc[1]['count'],
        title = {"text": f"Second Category: {top_3_cat.index[1]} <br> with total Papers: "},
        domain = {'x': [0, 0.33], 'y': [0.2, 0.8]}))

    fig.add_trace(go.Indicator(
        value = top_3_cat.iloc[0]['count'],
        title = {"text": f"First Category: {top_3_cat.index[0]} <br> with total Papers: "},
        domain = {'x': [0.33, 0.67], 'y': [0.4, 1]}))

    fig.add_trace(go.Indicator(
        value = top_3_cat.iloc[2]['count'],
        title = {"text": f"Third Category: {top_3_cat.index[2]} <br> with total Papers: "},
        domain = {'x': [0.67, 1], 'y': [0, 0.6]}))

    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1 :
        input_year = st.multiselect('Year', options=year_list + ['All'], default='All')
        if "All" in input_year:
            input_year = year_list

    with col2 :
        input_cat = st.multiselect('Category', options=category_list + ['All'], default='All')
        if "All" in input_cat:
            input_cat = category_list

    grouped_bar_filtered = px.bar(
        num_paper_cat_year.loc[(num_paper_cat_year['year'].isin(input_year)) & (num_paper_cat_year['class'].isin(input_cat))],
        x="year",
        y="count",
        color='class',
        barmode='group',    
        title=f"Number of Papers for each Category by Year",
        labels={"year": "Year", "count": "Number of Papers", "class": "Category"},
    )
    st.plotly_chart(grouped_bar_filtered, use_container_width=True)

    # --- Second Visualization: Group Authors by Number of Papers ---
    st.subheader("2. Top K Publisher by Number of Papers")

    publisher_grouped = papers_df.groupby('publishername').size().reset_index(name='count')

    top_k = st.slider("Top K Publisher", min_value=1, max_value=10, value=10)

    # All groups
    publisher_bar_all = px.bar(
        publisher_grouped.nlargest(n=top_k, columns='count'),
        x="publishername",
        y="count",
        title=f"Top {top_k} Publisher by Number of Papers",
    )
    st.plotly_chart(publisher_bar_all, use_container_width=True)

    

#=============================================================================
    affiliations_df_ll = affiliations_df[(affiliations_df['Latitude'] != 'N/A') & (affiliations_df['Longitude'] != 'N/A')]
    st.subheader("3. Analysis of Paper Density by Geographical Location")
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        data=affiliations_df_ll,
        get_position=['Longitude', 'Latitude'],
        get_weight='paper_count',
        opacity=0.7,
        radius_pixels=50,
        threshold=0.1
    )

    # Define the deck.gl view (for zoom and center positioning)
    view = pdk.ViewState(
        latitude=affiliations_df_ll['Latitude'].mean(),
        longitude=affiliations_df_ll['Longitude'].mean(),
        zoom=1,
        pitch=0
    )

    # Create the Pydeck deck object
    deck = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view,
        map_style='mapbox://styles/mapbox/dark-v10',  # You can change the map style if needed
        tooltip={"text": "{name} \n{Country}\n{City}"}
    )
    st.pydeck_chart(deck)