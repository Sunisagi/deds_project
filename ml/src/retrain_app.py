from flask import Flask, request, jsonify
from recommendation import make_model

app = Flask(__name__)

@app.route('/retrain', methods=['POST'])
def run_script():
    try:
        # Path to your script
        make_model(SET_RETRAIN=True)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
