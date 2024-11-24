from flask import Flask, request, jsonify

import nltk 
import numpy as np
import os
nltk.download("stopwords")
from nltk.corpus import stopwords
from recommendation import make_model,write_numpy_hadoop

stop_words = stopwords.words("english")
stop_words_array = np.array(stop_words)
os.makedirs("opt/temp", exist_ok=True)
np.save("opt/temp/stopwords_english.npy", stop_words_array)
write_numpy_hadoop(stop_words_array,"/temp/stopwords_english")


app = Flask(__name__)

@app.route('/retrain', methods=['POST'])
def run_script():
    try:
        # Path to your script
        # make_model(SET_RETRAIN=True)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
