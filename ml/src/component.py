import streamlit as st
import numpy as np
import pandas as pd
import numpy as np
from functools import partial
def create_table(df,show_meta=True):
    """
    Render a table using Streamlit components.
    """
    # Add index as rank starting from 1

    def click_button_paper(arg):
        # You can use the argument here to perform actions
        st.session_state.mode = "Paper Information"
        st.session_state.paper_id= arg  # Store the argument in session state
        st.session_state.author_id= None 


    df['rank'] = range(1, len(df) + 1)
    
    # Handle None (empty) fields
    

    for index, row in df.iterrows():
        # Assign a random light color for each row (simulating the original effect)
        light_color = np.random.randint(200, 256, size=3)
        light_color_css = f"rgb({light_color[0]}, {light_color[1]}, {light_color[2]})"
        # Create a container for each row
        with st.container():
            st.markdown(
                f"<div style='background-color: {light_color_css}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>",
                unsafe_allow_html=True,
            )

            score_value = row['score'] if 'score' in row else row['similarity_score']

            # Use the appropriate score value in the markdown
            st.markdown(
                f"**Rank {row['rank']}** | **Score:** {score_value}", unsafe_allow_html=True
            )

            # Create a layout with columns for structured display
            col1, col2,col3 = st.columns([1, 4, 1])
            col1.write("**Paper ID:**")
            col2.write(row["id"])
            
            col1, col2 = st.columns([1, 5])
            col1.write("**Title:**")
            col2.write(row["title"])

            col3.button("View Paper", key=f"view_paper_{row['id']}",on_click=partial(click_button_paper, row['id']))
                # # Store the paper_id in session state when the button is clicked
                # st.session_state["selected_paper_id"] = row["id"]
                # print(row['id'])
            

            if show_meta:
                with st.expander("View Abstract"):
                    st.write(row["description"])
                with st.expander("Meta Match on Query"):
                    col1, col2 = st.columns([1, 2])
                    col1.write("**Match Year:**")
                    col2.write(row["year"])
                    col1, col2 = st.columns([1, 2])
                    col1.write("**Match Country/City:**")
                    col2.write(row["country_or_city"])
                    col1, col2 = st.columns([1, 2])
                    col1.write("**Match Affiliation:**")
                    col2.write(row["affl"])

            st.markdown("</div>", unsafe_allow_html=True)

def create_recommendation_interface(model):
    """
    Create an interface for searching papers and generating recommendations.

    Parameters:
    - model: The recommendation model with a `get_recommendations` method.
    - create_table_func: A function that takes a DataFrame and returns HTML for rendering a table.
    """
    # Create columns
    col1, col2 = st.columns([2, 1])  # 2:1 ratio for width

    # Input fields
    text_input = col1.text_input("Search For Paper")
    number = col2.number_input("Papers", min_value=5, max_value=100, value=10)
    
    # Button
    generate_button = st.button("Generate Recommendations")

    if generate_button:
        if text_input:  # Check if there's any input
            st.write("You entered:", text_input)
            
            # Get recommendations
            data = model.get_recommendations(text_input, number)
            
            # Render table
            create_table(data)
        else:
            st.write("Please enter a query to generate recommendations.")


def create_author(df):
    """
    Render a grid-based table with styled rows and alternating colors using HTML and CSS,
    with buttons triggering actions via the `click_button` function.
    """
    def click_button_author(arg):
        """
        Function to handle button clicks for author actions.
        """
        st.session_state.mode = "Author Information"
        st.session_state.paper_id = None  # Reset paper_id
        st.session_state.author_id = arg

    # Handle None (empty) fields

    with st.container():
        col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])
        col1.markdown("**Given-name**")
        col2.markdown("**Initials**")
        col3.markdown("**Surname**")
        col4.markdown("**Indexed-name**")
        col5.markdown("**ACTION**")
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    # Render rows dynamically
    for index, row in df.iterrows():
        row_class = "cell alt" if index % 2 == 1 else "cell"  # Alternate row coloring

        # Create a new container for each row
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2,1,2,2,1])
            row_style = f"text-align: center; margin: auto;"  # CSS for centering content
            col1.markdown(f"<div class='{row_class}' >{row['given-name']}</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='{row_class}' >{row['initials']}</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='{row_class}' >{row['surname']}</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='{row_class}' >{row['indexed-name']}</div>", unsafe_allow_html=True)
            col5.button("View", key=f"view_author_{row['auid']}",on_click=partial(click_button_author, row['auid']))



def create_normal_table(df):
    """
    Render a table using Streamlit components.
    """
    # Add index as rank starting from 1

    def click_button_paper(arg):
        # You can use the argument here to perform actions
        st.session_state.mode = "Paper Information"
        st.session_state.paper_id= arg  # Store the argument in session state
        st.session_state.author_id= None 


    for index, row in df.iterrows():
        # Assign a random light color for each row (simulating the original effect)
        light_color = np.random.randint(200, 256, size=3)
        light_color_css = f"rgb({light_color[0]}, {light_color[1]}, {light_color[2]})"
        # Create a container for each row
        with st.container():
            st.markdown(
                f"<div style='background-color: {light_color_css}; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>",
                unsafe_allow_html=True,
            )

            col1, col2,col3 = st.columns([1, 4, 1])
            col1.write("**Paper ID:**")
            col2.write(row["id"])
            
            col1, col2 = st.columns([1, 5])
            col1.write("**Title:**")
            col2.write(row["title"])

            col3.button("View Paper", key=f"view_paper_{row['id']}",on_click=partial(click_button_paper, row['id']))
            st.markdown("</div>", unsafe_allow_html=True)