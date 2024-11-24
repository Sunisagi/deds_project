import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk 
import pandas as pd
from difflib import SequenceMatcher
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import json
import os
from tqdm import tqdm
import pickle
from connection_setup import set_classpath
from pyarrow import fs
import io
import gc
from torch.utils.data import DataLoader, Dataset


# Read a JSON file from Hadoop
def read_json_hadoop(file_name,return_json = False):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    with hdfs.open_input_stream(file_name) as f:
        content = f.read().decode("utf-8")  
        content_io = io.StringIO(content)
        if return_json:
            return json.load(content_io) 
        else:
            return pd.read_json(content_io, orient='records', lines=True)  

def read_numpy_hadoop(file_name):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)

    # Read the file from HDFS into memory
    with hdfs.open_input_stream(file_name) as file:
        data = file.read()

    # Deserialize the data into a NumPy array
    with io.BytesIO(data) as buffer:
        array = np.load(buffer)

    return array

# List all files in a folder on Hadoop
def ls_hadoop(folder):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    file_names = hdfs.get_file_info(fs.FileSelector(folder, recursive=True))
    return file_names

# Find the latest file based on the prefix and date in the filename
def find_latest_file_for_prefix(prefix, file_names):
    # Filter files with the given prefix
    filtered_files = [f for f in file_names if f.startswith(prefix)]
    if not filtered_files:
        print("No files found for the given prefix.")
        return None

    # Sort files by date extracted from the filename
    filtered_files.sort(
        key=lambda x: datetime.strptime("_".join(x.split("_")[-3:]).split(".")[0], "%Y_%m_%d")
    )

    # Return the latest file
    return filtered_files[-1] if filtered_files else None

# Function to load the latest file based on prefix and folder
def load_latest_numpy(prefix, folder):
    file_names = [f.base_name for f in ls_hadoop(folder)]
    file_name = find_latest_file_for_prefix(prefix, file_names)
    if file_name is None:
        return None

    print(f"Load File: {folder}/{file_name}")
    return read_numpy_hadoop(f"{folder}/{file_name}")

# Additional functions for specific file types
def load_latest_json(prefix, folder):
    file_names = [f.base_name for f in ls_hadoop(folder)]
    file_name = find_latest_file_for_prefix(prefix, file_names)
    if file_name is None:
        return None

    print(f"Load File: {folder}/{file_name}")
    return read_json_hadoop(f"{folder}/{file_name}", return_json=True)

def load_latest_file(prefix, folder):
    file_names = [f.base_name for f in ls_hadoop(folder)]
    file_name = find_latest_file_for_prefix(prefix, file_names)
    if file_name is None:
        return None

    print(f"Load File: {folder}/{file_name}")
    return read_json_hadoop(f"{folder}/{file_name}")

def load_latest_pkl(prefix, folder):
    file_names = [f.base_name for f in ls_hadoop(folder)]
    file_name = find_latest_file_for_prefix(prefix, file_names)
    if file_name is None:
        return None

    print(f"Load File: {folder}/{file_name}")
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    with hdfs.open_input_stream(f"{folder}/{file_name}") as f:
        return pickle.load(f)

def write_json_hadoop(data,file_name) :
    set_classpath()
    json_data = json.dumps(data)
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    with hdfs.open_output_stream(f'{file_name}.json') as file :
        file.write(json_data.encode('utf-8'))

def write_numpy_hadoop(data, file_name):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)

    # Serialize the NumPy data into a bytes stream
    with io.BytesIO() as buffer:
        np.save(buffer, data)
        buffer.seek(0)  # Reset buffer position to the beginning
        with hdfs.open_output_stream(f'{file_name}.npy') as file:
            file.write(buffer.read())

def write_pickle_hadoop(data, file_name):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)  # Adjust with your Hadoop configuration
    with hdfs.open_output_stream(f"{file_name}.pkl") as file:
        pickle.dump(data, file)


    
def create_text_similarity_json(paper_df, model, tokenizer, save_dir):
    date_str = datetime.now().strftime("%Y_%m_%d")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Function to generate embeddings with specified max length
    def get_embedding(text, max_length):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        return embeddings.squeeze().numpy()  # Keep as NumPy array
    
    # Initialize lists to store embeddings and indices
    title_embeddings = []
    abstract_embeddings = []
    index_to_id = []

    # Generate embeddings for titles and save them to arrays
    for _, row in tqdm(paper_df.iterrows(), desc="Generating Embeddings", total=len(paper_df)):
        paper_id = row['id']
        index_to_id.append(paper_id)  # Maintain index-to-id mapping
        title_embedding = get_embedding(row['title-clean'], max_length=128)
        title_embeddings.append(title_embedding)

        abstract_embedding = get_embedding(row['description-clean'], max_length=512)
        abstract_embeddings.append(abstract_embedding)

    # Convert title embeddings to a NumPy array and save
    title_embeddings_array = np.array(title_embeddings)
    abstract_embeddings_array = np.array(abstract_embeddings)

    # print(title_embeddings_array[:5,:5])
    # print(abstract_embeddings_array[:5,:5])
    
    write_numpy_hadoop(title_embeddings_array, f"{save_dir}/title_embeddings_{date_str}")
    write_numpy_hadoop(abstract_embeddings_array, f"{save_dir}/abstract_embeddings_{date_str}")
    write_numpy_hadoop(np.array(index_to_id), f"{save_dir}/index_to_id_{date_str}")

    del title_embeddings_array
    del abstract_embeddings_array
    del index_to_id
    # Run garbage collector
    gc.collect()
    print("Embeddings and index-to-id mapping saved successfully.")

class TextRecommender:
    def __init__(self, model, tokenizer, paper_df,id_to_index, title_embeddings, abstract_embeddings, affiliations_df, top_n=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.paper_df = paper_df
        self.affiliations_df = affiliations_df
        self.top_n = top_n
        self.stop_words = read_numpy_hadoop("/temp/stopwords_english.npy")

        paper_indices = []
        for idx, paper in paper_df.iterrows():
            paper_id = paper['id']  # Extract the paper_id from the current row
            paper_indices.append(id_to_index[paper_id])  # Get the index from id_to_index and append it

        self.title_matrix = title_embeddings[paper_indices]
        self.abstract_matrix = abstract_embeddings[paper_indices]

        # Prepare city and country terms for filtering
        self.country_or_city = affiliations_df.city.unique().tolist() + affiliations_df.country.unique().tolist()
        self.country_or_city = [str(cc).lower() for cc in self.country_or_city]

        # Generate vocabulary-filtered affiliation words
        self.vocabs = self.tokenizer.get_vocab()
        affiliation_words = []
        for _, row in self.affiliations_df.iterrows():
            ls = re.split(r"[ ,\-]+", row["name"].lower())
            for word in ls:
                if word not in self.vocabs and word not in affiliation_words and word not in self.country_or_city:
                    affiliation_words.append(word)
        self.pattern_name = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in affiliation_words + self.country_or_city) + r')\b', re.IGNORECASE)
    
    def text_preprocessing(self,s):
        s = s.lower()
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\b(1[0-9]{3}|2[0-9]{3})\b", "", s)
        s = re.sub(r'([©\'\"\(\)\!\\])', r' ', s)
        s = " ".join([word for word in s.split(' ') if word not in self.stop_words])
        s = self.pattern_name.sub("", s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def remove_unk_tokens(self,query_sentence):
        vocabs = self.tokenizer.get_vocab()
        keep = []
        unique_word = []
        for word in re.split(r"[ ,\-]+", query_sentence.lower()):
            if word in vocabs:
                keep.append(word)
            else:
                unique_word.append(word)
        
        # Rejoin the tokens into a clean sentence
        clean_sentence = ' '.join(keep)
        return clean_sentence, unique_word

    def remove_city_country(self,query_sentence):
        # Remove any tokens that are classified as "unk" by SciBERT
        keep = []
        unique_word = []
        for word in re.split(r"[ ,\-]+", query_sentence.lower()):
            if word not in self.country_or_city:
                keep.append(word)
            else:
                unique_word.append(word)
        
        # Rejoin the tokens into a clean sentence
        clean_sentence = ' '.join(keep)
        return clean_sentence, unique_word
    
    def remove_years(self, query_sentence):
        """Remove tokens that match a year pattern from the query sentence."""
        keep = []
        extracted_years = []
        
        for word in re.split(r"[ ,\-]+", query_sentence.lower()):
            # Match tokens that look like years (4-digit numbers starting with 19 or 20)
            if re.match(r'\b(19|20)\d{2}\b', word):
                extracted_years.append(word)
            else:
                keep.append(word)
        
        # Rejoin the remaining tokens into a clean sentence
        clean_sentence = ' '.join(keep)
        return clean_sentence, extracted_years

    def match_word(self, query, target_word):
        """
        Calculate similarity score between query and target word prefixes
        
        Args:
            query (str): Input query string
            target_word (str): The complete word to match against
            
        Returns:
            float: Returns highest similarity score found (between 0 and 1)
        """
        # Convert both strings to lowercase
        query = query.lower().strip()
        target_word = target_word.lower().strip()

            
        if abs(len(target_word) - len(query)) <= 2:
            return SequenceMatcher(None, query, target_word).ratio()
        return 0.0

    def find_matching_affiliations(self,query_words, threshold=0.8):
        matches = []
        seen_affids = set()  # Track unique affiliations
        
        for _, row in self.affiliations_df.iterrows():
            affiliation_name = row['name']
            max_word_score = 0  # Track highest score for this affiliation
            
            # Check each word in query against each word in affiliation
            for word in query_words:
                for affiliation_word in re.split(r"[ ,\-]+", affiliation_name.lower()):
                    word_match_score = self.match_word(word, affiliation_word)
                    max_word_score = max(max_word_score, word_match_score)
            # If score meets threshold and affiliation not seen yet
            if max_word_score >= threshold and row['affid'] not in seen_affids:
                matches.append({
                    'affid': row['affid'],
                    'name': row['name'],
                    'match_score': max_word_score,
                    'name_length': len(row['name'])  # Add name length for sorting
                })
                seen_affids.add(row['affid'])
        
        # Convert to DataFrame and sort
        matches_df = pd.DataFrame(matches)
        if not matches_df.empty:
            # Sort by score (descending) and name length (ascending)
            matches_df = matches_df.sort_values(
                by=['match_score', 'name_length'],
                ascending=[False, True]
            )
            # Drop the name_length column used for sorting
            matches_df = matches_df.drop('name_length', axis=1)
        
        return matches_df

    def preprocess_query(self, query):
        """Preprocess the query by removing unknown tokens and city/country names, then applying text processing."""
        sentence, year = self.remove_years(query.lower())
        sentence, cc = self.remove_city_country(sentence)
        sentence, unks = self.remove_unk_tokens(sentence)
        sentence = self.text_preprocessing(sentence)
        match_affiliations = self.find_matching_affiliations(unks)
        return sentence, cc, match_affiliations,year

    def get_query_embedding(self, query, max_length):
        """Get the embedding of the query text."""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        return embeddings.squeeze().numpy()

    def candidate_generation(self, sentence):
        """Generate candidate similarity scores based on title and abstract matrices."""
        title_query_embedding = self.get_query_embedding(sentence, max_length=128)
        abstract_query_embedding = self.get_query_embedding(sentence, max_length=512)
        
        # Calculate similarity scores for titles and abstracts
        title_similarity_scores = cosine_similarity([title_query_embedding], self.title_matrix).flatten()
        abstract_similarity_scores = cosine_similarity([abstract_query_embedding], self.abstract_matrix).flatten()

        # Combine similarity scores with weighting
        combined_similarity_scores = 0.7 * title_similarity_scores + 0.3 * abstract_similarity_scores
        return combined_similarity_scores

    def reranker(self, combined_similarity_scores, match_affiliations, cc, year):
        """
        Adjust similarity scores based on affiliation and city/country matches.
        Only check city/country if no affiliation matches are found.
        
        Args:
            combined_similarity_scores (np.array): Initial similarity scores
            match_affiliations (pd.DataFrame): Matching affiliations with scores
            cc (list): List of cities/countries to check
            
        Returns:
            np.array: Adjusted similarity scores
        """
        Q3 = np.quantile(combined_similarity_scores, 0.75)
        avg_similarity = combined_similarity_scores.mean()
        adjusted_scores = combined_similarity_scores.copy()
        
        for idx, paper in self.paper_df.iterrows():
            if combined_similarity_scores[idx] >= Q3:
                paper_affiliations = paper['affiliations']
                # Check affiliation matches first
                matched_affiliation_scores = [
                    row["match_score"] 
                    for _, row in match_affiliations.iterrows() 
                    if row["affid"] in paper_affiliations
                ]
                if matched_affiliation_scores:
                    # If we have affiliation matches, use those and skip cc checking
                    highest_affiliation_score = max(matched_affiliation_scores) - 0.8
                    adjusted_scores[idx] += avg_similarity * highest_affiliation_score

                
                elif cc:  # Only check cc if no affiliation matches and cc is not empty
                    relevant_affiliations = self.affiliations_df[
                        self.affiliations_df.affid.isin(paper_affiliations)
                    ]
                    # Check if any city or country matches
                    if any(
                        (relevant_affiliations['country'].str.lower().isin(cc)) |
                        (relevant_affiliations['city'].str.lower().isin(cc))
                    ):
                        adjusted_scores[idx] += avg_similarity *0.1
                if year and str(paper["year"]) in year:
                    adjusted_scores[idx] += 1
            
        return adjusted_scores

    def get_recommendations(self, query, N= None):
        """Generate recommendations based on the query."""
        if not N:
            N = self.top_n

        sentence, cc, match_affiliations,year = self.preprocess_query(query)
        combined_similarity_scores = self.candidate_generation(sentence)
        adjusted_scores = self.reranker(combined_similarity_scores, match_affiliations, cc, year)

        # Get top N most similar papers
        top_indices = np.argsort(adjusted_scores)[-N:][::-1]
        row_score = adjusted_scores[top_indices]
        recommendations = self.paper_df.iloc[top_indices][['id', 'title', 'description', 'affiliations','year']]

        # Add matched affiliations and city/country matches
        matched_affiliations = []
        matched_cities_or_countries = []
        matched_years = [] 
        for _, row in recommendations.iterrows():
            paper_affiliations = row['affiliations']

            # Affiliation names
            matched_names = [
                affiliation_row["name"] for _, affiliation_row in match_affiliations.iterrows()
                if affiliation_row["affid"] in paper_affiliations
            ]
            matched_affiliations.append(", ".join(matched_names) if matched_names else "none")

            # City or country matches, looping through each affiliation in paper_affiliations
            supplemental_matches = []
            for paper_affid in paper_affiliations:
                # Find the affiliation row based on affidavit
                affiliation_row = self.affiliations_df.loc[self.affiliations_df['affid'] == paper_affid].iloc[0]

                # Check city/country matches for this affiliation
                for city_country in cc:
                    if (str(affiliation_row['country']).lower() == city_country.lower()) or \
                    (str(affiliation_row['city']).lower() == city_country.lower()):
                        supplemental_matches.append(city_country)

            if str(row["year"]) in year:
                matched_years.append(str(row["year"]))
            else:
                matched_years.append("none")

            # Append the matched cities or countries
            matched_cities_or_countries.append(", ".join(supplemental_matches) if supplemental_matches else "none")

        # Add `affl` and `country_or_city` columns to the recommendations DataFrame
        recommendations = recommendations.assign(affl=matched_affiliations, country_or_city=matched_cities_or_countries,year=matched_years,score = row_score)
        return recommendations
    
    def find_similar_papers(self, paper_id, N=None):
        """Find the top N similar papers to the given paper_id, using a 0.5 weight for title and 0.5 for abstract similarity."""
        if not N:
            N = self.top_n
        paper_id = str(paper_id)
        # Check if paper_id exists in the DataFrame
        if paper_id not in self.paper_df['id'].values:
            print(f"Paper ID {paper_id} does not exist in the dataset.")
            return None, None
        
        # Get the index of the paper in the DataFrame
        paper = self.paper_df[self.paper_df['id'] == paper_id]
        paper_idx = paper.index[0]

        def get_row_cosine_similarity(matrix, row_index):
            row_vector = matrix[row_index].reshape(1, -1)  # Reshape to 2D array
            similarity = cosine_similarity(row_vector, matrix)  # Compute similarity
            return similarity.flatten()  # Flatten the result to a 1D array

        # Example usage:
        title_similarity_for_row = get_row_cosine_similarity(self.title_matrix, 0)  # Row 0
        abstract_similarity_for_row = get_row_cosine_similarity(self.abstract_matrix, 0)
        
        # Calculate combined similarity scores for titles and abstracts
        combined_similarity = 0.5 * title_similarity_for_row + 0.5 * abstract_similarity_for_row

        # Sort by similarity score and get the indices of the top N similar papers, excluding the input paper itself
        similar_indices = np.argsort(combined_similarity)[::-1]
        top_indices = [i for i in similar_indices if i != paper_idx][:N]

        # Retrieve top N similar papers' details
        similar_papers = self.paper_df.iloc[top_indices][['id', 'title', 'description']]
        similar_papers['similarity_score'] = combined_similarity[top_indices]

        return paper, similar_papers
    


    def get_author_similarity(self, author_id1, author_id2):
        """Get similarity score between two specific authors"""
        if author_id1 not in self.author_df['id'].values or author_id2 not in self.author_df['id'].values:
            return None
            
        idx1 = self.author_df[self.author_df['id'] == author_id1].index[0]
        idx2 = self.author_df[self.author_df['id'] == author_id2].index[0]
        
        return float(self.similarity_matrix[idx1, idx2])

def make_model(SET_RETRAIN  = False):
    embedding_dir ="/model/embedding"
    recommender_dir ="/model/recommender"

    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    papers_df = load_latest_file(f"clean_paper","/process")
    papers_df.reset_index(drop=True, inplace=True)
    affiliations_df = load_latest_file(f"affiliation","/process")
    affiliations_df.reset_index(drop=True, inplace=True)

    # Convert specific columns to strings
    if papers_df is not None:
        papers_df['id'] = papers_df['id'].astype(str)

    if affiliations_df is not None:
        affiliations_df['affid'] = affiliations_df['affid'].astype(str)

    tqdm.pandas()
    # Main logic for setting up the similarity matrices
    if SET_RETRAIN:
        create_text_similarity_json(papers_df, model, tokenizer,embedding_dir)

    # Load title and abstract embeddings from JSON
    title_embeddings = load_latest_numpy("title_embeddings",embedding_dir)
    abstract_embeddings = load_latest_numpy("abstract_embeddings",embedding_dir)
    index_to_id = load_latest_numpy("index_to_id",embedding_dir)
    id_to_index = {paper_id: idx for idx, paper_id in enumerate(index_to_id)}

    print("Loaded title and abstract similarity matrices.")

    recommender = TextRecommender(model, tokenizer, papers_df,id_to_index, title_embeddings, abstract_embeddings, affiliations_df, top_n=20)
    print("Finish")
    return recommender 

def save_model(recommender):
    date_str = datetime.now().strftime("%Y_%m_%d")
    recommender_dir = f"/model/recommender"
    write_pickle_hadoop(recommender, f"{recommender_dir}/model_{date_str}")


def load_model():
    recommender_dir = "/model/recommender"
    return load_latest_pkl("model",recommender_dir)

if __name__=="__main__":
    model = make_model(True)
    save_model(model)
    # print("start")
    # recommender_dir = "/model/recommender"
    # print(model.get_recommendations("brain damage 2018"))