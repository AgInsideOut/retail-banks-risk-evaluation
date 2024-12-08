# Retail Banks Risk Evaluation Model API

API service for credit risk prediction model deployment on Google Cloud Run. This service provides endpoints for credit risk assessment based on customer data.

## Repository Structure

```bash
credit-risk-model-api/
├── .gitignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py           # Flask application with API endpoints
│   ├── schemas.py        # Input/output validation schemas
│   └── utils.py          # Preprocessing and prediction helper functions
├── models/
│   ├── model.joblib      # Saved prediction model
│   └── woe_encoder.joblib # Weight of Evidence encoder
├── tests/
│   ├── __init__.py
│   ├── test_api.py       # API endpoint tests
│   └── test_model.py     # Model prediction tests
├── Dockerfile
├── requirements.txt
└── config.py             # Configuration settings
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run locally:
```bash
python -m flask run
```


## API Endpoints

### Health Check
```http
GET /health

Response:
{
    "status": "healthy",
    "model_loaded": true,
    "version": "1.0.0"
}
```
Used to verify service status and model availability.

### Predict
```http
POST /predict
Content-Type: application/json

Request Body:
{
    "features": {
        "AMT_INCOME_TOTAL": [135000.0],
        "AMT_CREDIT": [513000.0],
        "AMT_ANNUITY": [20000.0],
        "DAYS_BIRTH": [-10000],
        "DAYS_EMPLOYED": [-1500],
        ...
    }
}

Response:
{
    "status": "success",
    "prediction": {
        "default_probability": 0.23,
        "risk_category": "Medium Risk"
    }
}
```
Provides credit risk assessment based on customer features.

## Testing

Run tests:
```bash
python -m pytest
```

The test suite includes:
- API endpoint functionality tests
- Model prediction accuracy tests
- Input validation tests
- Preprocessing pipeline tests

## Configuration

Environment variables:
- `MODEL_PATH`: Path to saved model file (default: 'models/model.joblib')
- `DEBUG`: Enable debug mode (default: False)
- `PORT`: Server port (default: 8080)

## Feature Requirements

The prediction endpoint expects the following feature groups:
- Binary features (34 features including FLAG_OWN_CAR, FLAG_OWN_REALTY, etc.)
- Numerical features (70 features including AMT_INCOME_TOTAL, DAYS_BIRTH, etc.)
- Categorical features (14 features including NAME_CONTRACT_TYPE, CODE_GENDER, etc.)

See documentation for complete feature list and requirements.

## Error Handling

The API includes comprehensive error handling for:
- Invalid input data format
- Missing required features
- Model prediction errors
- Server errors
