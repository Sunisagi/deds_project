import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ast
import re
import json
from datetime import datetime
import io
from pathlib import Path
from connection_setup import set_classpath
from pyarrow import fs
from text_processing import find_value_by_key



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
def write_json_hadoop(json_data,file_name) :
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    with hdfs.open_output_stream(f'{file_name}.json') as file :
        file.write(json_data.encode('utf-8'))

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
def ls_hadoop(folder):
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)
    file_names = hdfs.get_file_info(fs.FileSelector(folder, recursive=True))
    return file_names
# Function to get the latest file based on date in the filename
def load_latest_file(prefix,folder):
    file_names = [f.base_name for f in ls_hadoop(folder)]
    file_name = find_latest_file_for_prefix(prefix, file_names)
    print(f"Load File :{f'{folder}/{file_name}'}")
    return read_json_hadoop(f'{folder}/{file_name}')

def find_latest_file_for_prefix(prefix, file_names):
    latest_file = None
    latest_date = None
    
    for file_name in file_names:
        # Check if the file name starts with the prefix
        if file_name.startswith(prefix + "_"):
            # Extract the date part from the file name
            date_str = file_name.replace(prefix + "_", "").replace(".json", "")
            
            # Parse the date
            date = datetime.strptime(date_str, "%Y_%m_%d")
            
            # Check if this is the latest date
            if latest_date is None or date > latest_date:
                latest_date = date
                latest_file = file_name
    
    return latest_file

def extract_feature(authors_df,affiliations_df):
    folder = '/scraping'
    new_author = 1
    new_affiliation = 1
    new_paper = []
    processed_paper_ids = set() 
    for f in ls_hadoop(folder):
        print(f.base_name)
        json_data = read_json_hadoop(f'{folder}/{f.base_name}',True)
        if find_value_by_key(json_data, "entry"):
            for data in find_value_by_key(json_data, "entry"):
                paper_id = data["id"].split("/")[-1]

                if paper_id in processed_paper_ids:
                    print(f"Skipping already processed paper: {paper_id}")
                    continue

                authors = []
                paper_affs = []
                if not isinstance(data["author"], list):
                    aus = [data["author"]]
                else:
                    aus = data["author"]
                for name in aus:
                    # Handle "name" and "surname" or "indexed-name"
                    if "." in name["name"].split()[-1] and len(name["name"].split()) >1:  # Handle the case where initials are in "name"
                        indexed_name = name["name"]
                        surname = indexed_name.split()[0]
                        initials = indexed_name.split()[1]
                        given_name = None
                    else:  # Handle the standard case
                        given_name, surname = name["name"].split()[0], name["name"].split()[-1]
                        initials = f"{given_name[0]}."
                        indexed_name = f"{surname} {given_name[0]}."

                    existing = authors_df[
                        (authors_df["given-name"] == given_name) & (authors_df["surname"] == surname)
                    ]
                    if existing.empty:
                        affiliations = []
                        if "arxiv:affiliation" in name:
                            if not isinstance(name["arxiv:affiliation"], list):
                                affs = [name["arxiv:affiliation"]["#text"]]
                            else:
                                affs = [aff["#text"] for aff in name["arxiv:affiliation"]]
                            for aff_name in affs:
                                # Check if any name in the DataFrame matches aff_name (case-insensitively)
                                match_found = False
                                for _, row in affiliations_df.iterrows():
                                    if "INFN" in aff_name or "Academy of Science" in aff_name:
                                        if aff_name == row["name"]:
                                            match_found = True
                                            matched_affid = row["affid"]
                                            # print(f"Match found for '{aff_name}': {row['name']}")
                                            break
                                    elif aff_name in row["name"]:
                                        if len(aff_name) > len(row["name"]) / 2:
                                            match_found = True
                                            matched_affid = row["affid"]
                                            break
                                    elif row["name"] in aff_name:
                                        if len(aff_name) / 2 < len(row["name"]):
                                            matched_affid = row["affid"]
                                            match_found = True
                                            break

                                if not match_found:
                                    # If no sufficient match, add a new row to the DataFrame
                                    new_aff_id = f"S{new_affiliation:07d}"
                                    new_affiliation += 1
                                    new_row = {"affid": new_aff_id, "name": aff_name}
                                    affiliations_df = pd.concat([affiliations_df, pd.DataFrame([new_row])], ignore_index=True)
                                    matched_affid = new_aff_id
                                affiliations.append(matched_affid)
                                if matched_affid not in paper_affs:
                                    paper_affs.append(matched_affid)

                        new_id = f"S{new_author:09d}"
                        new_row = pd.DataFrame({
                            "auid": [new_id],
                            "given-name": [given_name],
                            "surname": [surname],
                            "initials": [initials],
                            "indexed-name": [indexed_name],
                            "paper": [[paper_id]],
                            "affiliation": [affiliations],
                        })
                        authors_df = pd.concat([authors_df, new_row], ignore_index=True)
                        new_author += 1
                    else:
                        new_id = existing["auid"].iloc[0]
                        idx = existing.index[0]
                        authors_df.at[idx, "paper"] = authors_df.at[idx, "paper"] + [paper_id]
                    authors.append(str(new_id))
                new_row = pd.DataFrame({
                    "id": [paper_id],
                    "title": [data["title"].replace("\n", " ")],
                    "description": [data["summary"].replace("\n", " ")],
                    "date": [data["published"]],
                    "year": [data["published"].split("-")[0]],
                    "authors": [authors],
                    "affiliations": [paper_affs],
                })
                new_paper.append(new_row)
                processed_paper_ids.add(paper_id)
    
    new_paper_df = pd.concat(new_paper, ignore_index=True)
    
    return new_paper_df,authors_df,affiliations_df



def clean_text(papers_df,affiliations_df):
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    stop_words = read_numpy_hadoop("/temp/stopwords_english.npy").tolist()

    # papers_df = load_latest_file('paper','/process')
    # affiliations_df = load_latest_file('affiliation','/process')


    vocabs = tokenizer.get_vocab()
    affiliation_words = []
    country_or_city =affiliations_df.city.unique().tolist() + affiliations_df.country.unique().tolist()
    country_or_city = [str(cc).lower() for cc in country_or_city]
    for _,row in affiliations_df.iterrows():
        ls = re.split(r"[ ,\-]+", row["name"].lower())
        for word in ls:
            if word not in vocabs and word not in affiliation_words and word not in country_or_city:
                affiliation_words.append(word)
    
    pattern_name = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in affiliation_words+country_or_city) + r')\b', re.IGNORECASE)

    def text_preprocessing(s):
        """
        - Lowercase the sentence
        - Change "'t" to "not"
        - Remove "@name"
        - Isolate and remove punctuations except "?"
        - Remove other special characters
        - Remove stop words except "not" and "can"
        - Remove trailing whitespace
        """
        
        s = s.lower()
        s = re.sub(r"n\'t", " not", s)
        s = re.sub(r"\b(1[0-9]{3}|2[0-9]{3})\b", "", s)
        s = re.sub(r'([©\'\"\(\)\!\\])', r' ', s)
        s = " ".join([word for word in s.split(' ') if word not in stop_words])
        s = pattern_name.sub("", s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    papers_df['title-clean'] = papers_df['title'].apply(text_preprocessing)
    print("Title Are Cleaned")
    papers_df['description-clean'] = papers_df['description'].apply(text_preprocessing)
    print("Abstract Are Cleaned")

    return papers_df

def initialize_clean():
    date_str = datetime.now().strftime("%Y_%m_%d")
    papers_df = load_latest_file('paper','/json')
    affiliations_df = load_latest_file('affiliation','/json')
    clean_papers_df = clean_text(papers_df,affiliations_df)
    path = f"/json/clean_paper_{date_str}"
    write_json_hadoop(clean_papers_df.to_json(orient="records", lines=True),path)

def clean_scarpe():
    date_str = datetime.now().strftime("%Y_%m_%d")
    save_dir = "/process"
    authors_df = load_latest_file('author','/json')
    affiliations_df = load_latest_file('affiliation','/json')
    new_paper_df,authors_df,affiliations_df = extract_feature(authors_df,affiliations_df)
    # path = f"{save_dir}/paper_{date_str}"
    # write_json_hadoop(papers_df.to_json(orient="records", lines=True),path)
    path = f"{save_dir}/author_{date_str}"
    write_json_hadoop(authors_df.to_json(orient="records", lines=True),path)

    path = f"{save_dir}/affiliation_{date_str}"
    write_json_hadoop(affiliations_df.to_json(orient="records", lines=True),path)
    
    papers_df = load_latest_file('clean_paper','/json')
    print(len(new_paper_df))
    new_paper_df = clean_text(new_paper_df,affiliations_df)
    papers_df = pd.concat([papers_df,new_paper_df], ignore_index=True)

    path = f"{save_dir}/clean_paper_{date_str}"
    write_json_hadoop(papers_df.to_json(orient="records", lines=True),path)

    papers_df = papers_df.drop(columns=["title-clean", "description-clean"])
    path = f"{save_dir}/paper_{date_str}"
    write_json_hadoop(papers_df.to_json(orient="records", lines=True),path)

    print("Finish")


if __name__=="__main__":
    clean_scarpe()
    
