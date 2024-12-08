from flask import Flask, request, jsonify
from app.utils import preprocess_features, format_prediction
from app.schemas import validate_input
import joblib
from config import Config
from pathlib import Path

app = Flask(__name__)

model = joblib.load(Path(Config.MODEL_PATH))

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify service status.
    Returns:
        JSON response indicating service health status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.
    Returns:
        JSON response with prediction results
    """
    try:
        # Validate input
        data = request.get_json()
        if not validate_input(data):
            return jsonify({
                'error': 'Invalid input format',
                'status': 'error'
            }), 400

        # Preprocess features
        features = preprocess_features(data['features'])
        
        # Make prediction
        prediction = model.predict_proba(features)
        
        # Format response
        response = format_prediction(prediction)
        
        return jsonify({
            'prediction': response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG
    )