from flask import Flask, request, jsonify

import nltk 
import numpy as np
import os
nltk.download("stopwords")
from nltk.corpus import stopwords
from recommendation import make_model,write_numpy_hadoop,save_model
from text_processing import initialize
from text_clean import initialize_clean,clean_scarpe
from connection_setup import set_classpath
from pyarrow import fs


def path_exists_hadoop(path):
    """
    Check if a given path exists in the Hadoop file system.

    Args:
        path (str): The HDFS path to check.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    set_classpath()
    hdfs = fs.HadoopFileSystem("namenode", 8020)  # Replace "namenode" and port as necessary
    try:
        # Get file info for the path
        file_info = hdfs.get_file_info([path])
        # Check if the path exists
        return file_info[0].type != fs.FileType.NotFound
    except Exception as e:
        print(f"Error checking path {path}: {e}")
        return False
    


app = Flask(__name__)


stop_words = stopwords.words("english")
stop_words_array = np.array(stop_words)
os.makedirs("opt/temp", exist_ok=True)
np.save("opt/temp/stopwords_english.npy", stop_words_array)
write_numpy_hadoop(stop_words_array,"/temp/stopwords_english")
if not path_exists_hadoop("/json"):
    print('START')
    initialize()
    print('FINSIH PROCESSING')
    initialize_clean()
    print('FINSIH CLEANING')
else:
    print("PASS")

@app.route('/retrain', methods=['POST'])
def run_script():
    try:
        # Path to your script
        clean_scarpe()
        print("Finish Clean Scrape")
        model = make_model(True)
        save_model(model)
        print("Finish Save Model")
        
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
