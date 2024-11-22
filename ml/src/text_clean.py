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
from connection_setup import set_classpath
from pyarrow import fs


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
        if return_json:
            return json.loads(content) 
        else:
            return pd.read_json(content, orient='records') 
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

def extract_feature():
    folder = '/user/airflow/json'
    for f in ls_hadoop(folder):
        print(f.base_name)
        data = read_json_hadoop(f'{folder}/{f.base_name}',True)
        print(data["feed"]["title"])



# def initialize():
#     date_str = datetime.now().strftime("%Y_%m_%d")
#     nltk.download("stopwords")
#     from nltk.corpus import stopwords
    
#     model_name = "allenai/scibert_scivocab_uncased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)

#     papers_df = load_latest_file('paper','/json')
#     affiliations_df = load_latest_file('affiliation','/json')


#     vocabs = tokenizer.get_vocab()
#     affiliation_words = []
#     country_or_city =affiliations_df.city.unique().tolist() + affiliations_df.country.unique().tolist()
#     country_or_city = [str(cc).lower() for cc in country_or_city]
#     for _,row in affiliations_df.iterrows():
#         ls = re.split(r"[ ,\-]+", row["name"].lower())
#         for word in ls:
#             if word not in vocabs and word not in affiliation_words and word not in country_or_city:
#                 affiliation_words.append(word)
    
#     pattern_name = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in affiliation_words+country_or_city) + r')\b', re.IGNORECASE)

#     def text_preprocessing(s):
#         """
#         - Lowercase the sentence
#         - Change "'t" to "not"
#         - Remove "@name"
#         - Isolate and remove punctuations except "?"
#         - Remove other special characters
#         - Remove stop words except "not" and "can"
#         - Remove trailing whitespace
#         """
        
#         s = s.lower()
#         s = re.sub(r"n\'t", " not", s)
#         s = re.sub(r"\b(1[0-9]{3}|2[0-9]{3})\b", "", s)
#         s = re.sub(r'([©\'\"\(\)\!\\])', r' ', s)
#         s = " ".join([word for word in s.split(' ')
#         if word not in stopwords.words('english')])
#         s = pattern_name.sub("", s)
#         s = re.sub(r'\s+', ' ', s).strip()
#         return s
    
#     # Enable tqdm with pandas apply
#     tqdm.pandas()

#     # Apply text preprocessing with progress tracking
#     papers_df['title-clean'] = papers_df['title'].progress_apply(text_preprocessing)
#     papers_df['description-clean'] = papers_df['description'].progress_apply(text_preprocessing)
#     path = f"{PROJECT_ROOT}/data/process/clean_paper_{date_str}"
#     papers_df.to_csv(f'{path}.csv', index=False)
#     papers_df.to_json(f'{path}.json', orient="records", lines=True)


if __name__=="__main__":
    # initialize()
    papers_df = load_latest_file('paper','/json')
    authors_df = load_latest_file('author','/json')
    affiliations_df = load_latest_file('affiliation','/json')
    extract_feature()