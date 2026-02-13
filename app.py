from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [
            data["MedInc"],
            data["HouseAge"],
            data["AveRooms"],
            data["AveBedrms"],
            data["Population"],
            data["AveOccup"],
            data["Latitude"],
            data["Longitude"]
        ]
        prediction = model.predict([features])
        return jsonify({
            "predicted_price": float(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

