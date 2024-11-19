import os
import pandas as pd
import json
import ast
from datetime import datetime
import glob
from pathlib import Path

def find_value_by_key(data, target_key):
    """Recursively find the first value associated with a given key in the JSON object."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                return value  # Return the value if the key matches
            # Recur for the value and check if it returns a valid result
            found_value = find_value_by_key(value, target_key)
            if found_value is not None:
                return found_value  # Return found value if not None
    elif isinstance(data, list):
        for item in data:
            found_value = find_value_by_key(item, target_key)
            if found_value is not None:
                return found_value  # Return found value if not None

    return None

def find_values_by_key(data, target_key):
    """Recursively find all values associated with a given key in the JSON object."""
    found_values = []  # List to store found values

    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                found_values.append(value)  # Append the value if the key matches
            found_values.extend(find_values_by_key(value, target_key))  # Recur for the value
    elif isinstance(data, list):
        for item in data:
            found_values.extend(find_values_by_key(item, target_key))  # Recur for each item in the list

    return found_values  # Return the list of found values

def get_class(data):
    classifications = find_value_by_key(data,'classifications')
    target = []
    for cls in classifications:
        if cls['@type'] == 'SUBJABBR':
            if isinstance(cls['classification'], list):
                for c in cls['classification']:
                    target.append(c['$'])
            else:
                target = [cls['classification']]
    return target


def initialize():
    data_rows = []
    affiliations = {}
    citations = {}
    authors_name = {}

    current_path = Path.cwd()
    PROJECT_ROOT = current_path.parent 

    raw_path = f"{PROJECT_ROOT }/data/raw"
    date_str = datetime.now().strftime("%Y_%m_%d")
    save_dir = f"{PROJECT_ROOT }/data/process"
    os.makedirs(save_dir, exist_ok=True) 

    # Get a list of all folders in the main directory
    folders = [folder for folder in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, folder))]

    # Loop through each folder and process files within each
    for folder in folders:
        raw_folder = os.path.join(raw_path, folder)
        if folder.isdigit() and len(folder) == 4:
            # print(folder)
            for filename in os.listdir(raw_folder):
                affids = {}
                if filename.endswith('.json'):
                    file_path = f"{raw_folder}/{filename}"
                    
                    # Open and load the JSON data
                    with open(file_path, 'r', encoding='utf-8') as file:
                        json_data = json.load(file)
                    
                    # Extract values
                    title = find_value_by_key(json_data, 'dc:title')
                    description = find_value_by_key(json_data, 'dc:description')
                    publishername = find_value_by_key(json_data, 'publishername')
                    references = find_value_by_key(json_data, 'reference')
                    pdate = find_value_by_key(json_data, 'prism:coverDate')
                    url = find_value_by_key(json_data, 'prism:url')
                    # Handle nested key for copyright type
                    copyright_type = None
                    copyright_data = find_value_by_key(json_data, 'copyright')
                    if isinstance(copyright_data, dict):
                        copyright_type = copyright_data.get('@type')
                    
                    class_value = get_class(json_data)
                    
                    paper_id = url.split('/')[-1]
                    # Append a row to the data list
                    paper = {
                        'id': paper_id,
                        'title': title,
                        'description': description,
                        'publishername': publishername,
                        'copyright_type': copyright_type,
                        'date': pdate,
                        'year': pdate.split("-")[0],
                        'class': class_value,
                        'affiliations': []
                    }
                    cites = []

                    if isinstance(references, list):
                        for ref in references:
                            item_ids = ref.get("ref-info", {}).get("refd-itemidlist", {}).get("itemid", [])
                            
                            # Ensure item_ids is a list
                            if not isinstance(item_ids, list):
                                item_ids = [item_ids]
                            
                            # Find the itemid with "@idtype" of "SGR"
                            cite_id = None
                            for item in item_ids:
                                if item.get("@idtype") == "SGR":
                                    cite_id = item.get("$")
                                    break  # Stop once we find the SGR type
                            
                            if cite_id:
                                cites.append(cite_id)
                                
                                # Add citation information if cite_id exists
                                if cite_id not in citations:
                                    ref_authors = ref.get("ref-info", {}).get("ref-authors", {}).get("author", [])

                                    citations[cite_id] = {
                                        "ref-fulltext": ref.get("ref-fulltext", None),
                                        "ref-title": ref.get("ref-info", {}).get("ref-title", {}).get("ref-titletext", None),
                                        "ref-authors": [{
                                            "indexed-name" : ref_author["ce:indexed-name"],
                                            "seq" : ref_author["@seq"]
                                        } for ref_author in ref_authors],
                                        "ref-collaboration": ref.get("ref-info", {}).get("ref-authors", {}).get("collaboration", []),
                                        "paper_refer": []  # Initialize paper_refer as a list
                                    }
                                    
                                # Append filename to paper_refer list for this cite_id
                                if paper_id not in citations[cite_id]["paper_refer"]:
                                    citations[cite_id]["paper_refer"].append(paper_id)

                    elif isinstance(references, dict):
                        ref = references
                        item_ids = ref.get("ref-info", {}).get("refd-itemidlist", {}).get("itemid", [])
                        
                        if not isinstance(item_ids, list):
                            item_ids = [item_ids]
                        
                        cite_id = None
                        for item in item_ids:
                            if item.get("@idtype") == "SGR":
                                cite_id = item.get("$")
                                break
                        
                        if cite_id:
                            cites.append(cite_id)
                            
                            # Add citation information if cite_id exists
                            if cite_id not in citations:
                                ref_authors = ref.get("ref-info", {}).get("ref-authors", {}).get("author", [])

                                citations[cite_id] = {
                                    "ref-fulltext": ref.get("ref-fulltext", None),
                                    "ref-title": ref.get("ref-info", {}).get("ref-title", {}).get("ref-titletext", None),
                                    "ref-authors": [{
                                        "indexed-name" : ref_author["ce:indexed-name"],
                                        "seq" : ref_author["@seq"]
                                    } for ref_author in ref_authors],
                                    "ref-collaboration": ref.get("ref-info", {}).get("ref-authors", {}).get("collaboration", []),
                                    "paper_refer": []  # Initialize paper_refer as a list
                                }
                            
                            # Append filename to paper_refer list for this cite_id
                            if paper_id not in citations[cite_id]["paper_refer"]:
                                citations[cite_id]["paper_refer"].append(paper_id)
                    
                    affs = json_data.get("abstracts-retrieval-response",{}).get("affiliation",[])
                    if not isinstance(affs,list):
                        affs = [affs]
                    for aff in affs:
                        aff_id = aff.get("@id",None)
                        if aff_id:
                            # Check if the affiliation id exists, and if not, initialize the entry
                            if aff_id not in affiliations:
                                affiliations[aff_id] = {
                                    "city": aff.get("affiliation-city", None),
                                    "name": aff.get("affilname", None),
                                    "country": aff.get("affiliation-country", None),
                                    "paper_count": 0
                                }
                            else:
                                # Update the fields if they are None and the new entry has a non-None value
                                if affiliations[aff_id]["city"] is None and aff.get("affiliation-city"):
                                    affiliations[aff_id]["city"] = aff.get("affiliation-city")
                                
                                if affiliations[aff_id]["name"] is None and aff.get("affilname"):
                                    affiliations[aff_id]["name"] = aff.get("affilname")
                                
                                if affiliations[aff_id]["country"] is None and aff.get("affiliation-country"):
                                    affiliations[aff_id]["country"] = aff.get("affiliation-country")

                            affiliations[aff_id]["paper_count"] +=1
                            paper["affiliations"].append(aff_id)

                    if find_value_by_key(json_data, "author-group"):
                        author_group = find_value_by_key(json_data, "author-group")
                        if not isinstance(author_group, list):
                            author_group = [author_group]
                        aus = []
                        aus_seq = []
                        for author in author_group:
                            aff_id = author.get("affiliation",{}).get("@afid",None)
                            authors = author.get("author",[])
                            if not isinstance(authors,list):
                                authors = [authors]
                            for a in authors:
                                if a["@auid"] not in authors_name:
                                    authors_name[a["@auid"]] ={
                                        "given-name": a.get("ce:given-name",None),
                                        "initials": a.get("ce:initials",None),
                                        "surname": a.get("ce:surname",None),
                                        "indexed-name": a.get("ce:indexed-name",None),
                                        "affliation": aff_id,
                                        "paper" : [],
                                        # 'paper_seq':[]
                                    }
                                else:
                                    if authors_name[a["@auid"]]["given-name"] is None and a.get("ce:given-name"):
                                        authors_name[a["@auid"]]["given-name"] = a.get("ce:given-name")

                                    if authors_name[a["@auid"]]["initials"] is None and a.get("ce:initials"):
                                        authors_name[a["@auid"]]["initials"] = a.get("ce:initials")
                                
                                    if authors_name[a["@auid"]]["surname"] is None and a.get("ce:surname"):
                                        authors_name[a["@auid"]]["surname"] = a.get("ce:surname")
                                    
                                    if authors_name[a["@auid"]]["indexed-name"] is None and a.get("ce:indexed-name"):
                                        authors_name[a["@auid"]]["indexed-name"] = a.get("ce:indexed-name")

                                    if authors_name[a["@auid"]]["affliation"] is None and aff_id:
                                        authors_name[a["@auid"]]["affliation"] = str(aff_id)

                                if paper_id not in authors_name[a["@auid"]]["paper"]:
                                    authors_name[a["@auid"]]["paper"].append(paper_id)

                                if a["@auid"] not in aus:
                                    aus.append(a["@auid"])
                                
                                # authors_name[a["@auid"]]["paper_seq"].append({
                                #     "paper_id": paper_id, 
                                #     "seq": a.get("@seq",None)
                                # })
                                # aus_seq.append({
                                #     "auid": a["@auid"], 
                                #     "seq": a.get("@seq",None)
                                # })
                                    
                    paper["cites"] = cites
                    paper["authors"] = aus
                    # paper["authors_seq"] = aus_seq
                    data_rows.append(paper)
        else:
            continue
    # Create the dataframes
    papers_df = pd.DataFrame(data_rows)
    papers_df.dropna(inplace=True)

    citations_df = pd.DataFrame.from_dict(citations, orient='index').reset_index().rename(columns={'index': 'citeid'})
    author_df = pd.DataFrame.from_dict(authors_name, orient='index').reset_index().rename(columns={'index': 'auid'})
    affiliations_df = pd.DataFrame.from_dict(affiliations, orient='index').reset_index().rename(columns={'index': 'affid'})

    # Save the files with the date added to filenames
    path = f"{save_dir}/paper_{date_str}"
    papers_df.to_csv(f'{path}.csv', index=False)
    papers_df.to_json(f'{path}.json', orient="records", lines=True)

    path = f"{save_dir}/cite_{date_str}"
    citations_df.to_csv(f'{path}.csv', index=False)
    citations_df.to_json(f'{path}.json', orient="records", lines=True)

    path = f"{save_dir}/author_{date_str}"
    author_df.to_csv(f'{path}.csv', index=False)
    author_df.to_json(f'{path}.json', orient="records", lines=True)

    path = f"{save_dir}/affiliation_{date_str}"
    affiliations_df.to_csv(f'{path}.csv', index=False)
    affiliations_df.to_json(f'{path}.json', orient="records", lines=True)

def load_latest_file(prefix, extension="json"):
    # Get a list of files matching the pattern (e.g., "paper_*.json")
    files = glob.glob(f"{prefix}_*.{extension}")
    
    # If no files match, return None or handle as needed
    if not files:
        print("No files found.")
        return None
    
    # Sort files by date extracted from filename (assuming date is in format YYYY_MM_DD)
    files.sort(key=lambda x: datetime.strptime(x.split("_")[-3:], "%Y_%m_%d." + extension))
    
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