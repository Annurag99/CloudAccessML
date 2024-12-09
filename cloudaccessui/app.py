from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


app = Flask(__name__)

# Load the pre-trained model from the pkl file
with open('best_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route to render a simple HTML page
@app.route("/")

def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON for features
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)

        # Make a prediction
        prediction = model.predict(features).tolist()
        probability = model.predict_proba(features).tolist()

        # Return prediction and probability
        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
