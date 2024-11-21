import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Set, Tuple
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import ast
import re
import nltk
import json
from datetime import datetime
import glob
from pathlib import Path

# Function to get the latest file based on date in the filename
def load_latest_file(prefix, extension="json"):
    # Get a list of files matching the pattern (e.g., "paper_*.json")
    files = glob.glob(f"{prefix}_*.{extension}")
    
    # If no files match, return None or handle as needed
    if not files:
        print("No files found.")
        return None
    
    # Sort files by date extracted from filename (assuming date is in format YYYY_MM_DD)
    files.sort(key=lambda x: datetime.strptime("_".join(x.split("_")[-3:]).split(f".{extension}")[0], "%Y_%m_%d"))
    
    # Get the latest file (last in sorted list)
    latest_file = files[-1]
    print(f"Loading latest file: {latest_file}")
    
    # Load the file based on its extension
    if extension == "json":
        return pd.read_json(latest_file, orient="records", lines=True)
    elif extension == "csv":
        return pd.read_csv(latest_file)
    else:
        print("Unsupported file extension.")
        return None

def initialize():
    current_path = Path.cwd()  # or os.getcwd()
    PROJECT_ROOT = current_path.parent 

    date_str = datetime.now().strftime("%Y_%m_%d")
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    
    model_name = "allenai/scibert_scivocab_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    papers_df = load_latest_file(f"{PROJECT_ROOT}/data/process/paper","json")
    affiliations_df = load_latest_file(f"{PROJECT_ROOT}/data/process/affiliation","json")


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
        s = " ".join([word for word in s.split(' ')
        if word not in stopwords.words('english')])
        s = pattern_name.sub("", s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    # Enable tqdm with pandas apply
    tqdm.pandas()

    # Apply text preprocessing with progress tracking
    papers_df['title-clean'] = papers_df['title'].progress_apply(text_preprocessing)
    papers_df['description-clean'] = papers_df['description'].progress_apply(text_preprocessing)
    path = f"{PROJECT_ROOT}/data/process/clean_paper_{date_str}"
    papers_df.to_csv(f'{path}.csv', index=False)
    papers_df.to_json(f'{path}.json', orient="records", lines=True)