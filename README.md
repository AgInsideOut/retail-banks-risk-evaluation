# Retail Banks Risk Evaluation Model API

API service for credit risk prediction model deployment on Google Cloud Run. This service provides endpoints for credit risk assessment based on customer data.

## Repository Structure

```bash
credit-risk-model-api/
├── .gitignore
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── utils.py
├── models/
│   ├── model.joblib
│   └── woe_encoder.joblib
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── Dockerfile
├── requirements.txt
└── config.py
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
    "FLAG_OWN_CAR": "Y",
    "FLAG_OWN_REALTY": "Y",
    "FLAG_MOBIL": "Y",
    "FLAG_EMP_PHONE": "Y",
    "FLAG_WORK_PHONE": "Y",
    "FLAG_CONT_MOBILE": "Y",
    "FLAG_PHONE": "Y",
    "FLAG_EMAIL": "Y",
    "REG_REGION_NOT_LIVE_REGION": "N",
    "REG_REGION_NOT_WORK_REGION": "N",
    "LIVE_REGION_NOT_WORK_REGION": "N",
    "REG_CITY_NOT_LIVE_CITY": "N",
    "REG_CITY_NOT_WORK_CITY": "N",
    "LIVE_CITY_NOT_WORK_CITY": "N",
    "FLAG_DOCUMENT_2": "N",
    "FLAG_DOCUMENT_3": "Y",
    "FLAG_DOCUMENT_4": "N",
    "FLAG_DOCUMENT_5": "N",
    "FLAG_DOCUMENT_6": "N",
    "FLAG_DOCUMENT_7": "N",
    "FLAG_DOCUMENT_8": "N",
    "FLAG_DOCUMENT_9": "N",
    "FLAG_DOCUMENT_10": "N",
    "FLAG_DOCUMENT_11": "N",
    "FLAG_DOCUMENT_12": "N",
    "FLAG_DOCUMENT_13": "N",
    "FLAG_DOCUMENT_14": "N",
    "FLAG_DOCUMENT_15": "N",
    "FLAG_DOCUMENT_16": "N",
    "FLAG_DOCUMENT_17": "N",
    "FLAG_DOCUMENT_18": "N",
    "FLAG_DOCUMENT_19": "N",
    "FLAG_DOCUMENT_20": "N",
    "FLAG_DOCUMENT_21": "N",
    "CNT_CHILDREN": 1,
    "AMT_INCOME_TOTAL": 180000,
    "AMT_CREDIT": 600000,
    "AMT_ANNUITY": 30000,
    "AMT_GOODS_PRICE": 550000,
    "REGION_POPULATION_RELATIVE": 0.02,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -2000,
    "DAYS_REGISTRATION": -4000,
    "DAYS_ID_PUBLISH": -3000,
    "OWN_CAR_AGE": 5,
    "CNT_FAM_MEMBERS": 3,
    "REGION_RATING_CLIENT": 2,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "HOUR_APPR_PROCESS_START": 12,
    "EXT_SOURCE_1": 0.5,
    "EXT_SOURCE_2": 0.6,
    "EXT_SOURCE_3": 0.7,
    "APARTMENTS_AVG": 0.05,
    "BASEMENTAREA_AVG": 0.05,
    "YEARS_BEGINEXPLUATATION_AVG": 0.8,
    "YEARS_BUILD_AVG": 0.8,
    "COMMONAREA_AVG": 0.05,
    "ELEVATORS_AVG": 0.05,
    "ENTRANCES_AVG": 0.05,
    "FLOORSMAX_AVG": 0.05,
    "FLOORSMIN_AVG": 0.05,
    "LANDAREA_AVG": 0.05,
    "LIVINGAPARTMENTS_AVG": 0.05,
    "LIVINGAREA_AVG": 0.05,
    "NONLIVINGAPARTMENTS_AVG": 0.05,
    "NONLIVINGAREA_AVG": 0.05,
    "APARTMENTS_MODE": 0.05,
    "BASEMENTAREA_MODE": 0.05,
    "YEARS_BEGINEXPLUATATION_MODE": 0.8,
    "YEARS_BUILD_MODE": 0.8,
    "COMMONAREA_MODE": 0.05,
    "ELEVATORS_MODE": 0.05,
    "ENTRANCES_MODE": 0.05,
    "FLOORSMAX_MODE": 0.05,
    "FLOORSMIN_MODE": 0.05,
    "LANDAREA_MODE": 0.05,
    "LIVINGAPARTMENTS_MODE": 0.05,
    "LIVINGAREA_MODE": 0.05,
    "NONLIVINGAPARTMENTS_MODE": 0.05,
    "NONLIVINGAREA_MODE": 0.05,
    "APARTMENTS_MEDI": 0.05,
    "BASEMENTAREA_MEDI": 0.05,
    "YEARS_BEGINEXPLUATATION_MEDI": 0.8,
    "YEARS_BUILD_MEDI": 0.8,
    "COMMONAREA_MEDI": 0.05,
    "ELEVATORS_MEDI": 0.05,
    "ENTRANCES_MEDI": 0.05,
    "FLOORSMAX_MEDI": 0.05,
    "FLOORSMIN_MEDI": 0.05,
    "LANDAREA_MEDI": 0.05,
    "LIVINGAPARTMENTS_MEDI": 0.05,
    "LIVINGAREA_MEDI": 0.05,
    "NONLIVINGAPARTMENTS_MEDI": 0.05,
    "NONLIVINGAREA_MEDI": 0.05,
    "TOTALAREA_MODE": 0.05,
    "OBS_30_CNT_SOCIAL_CIRCLE": 1,
    "DEF_30_CNT_SOCIAL_CIRCLE": 0,
    "OBS_60_CNT_SOCIAL_CIRCLE": 1,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0,
    "DAYS_LAST_PHONE_CHANGE": -100,
    "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
    "AMT_REQ_CREDIT_BUREAU_DAY": 0,
    "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
    "AMT_REQ_CREDIT_BUREAU_MON": 1,
    "AMT_REQ_CREDIT_BUREAU_QRT": 1,
    "AMT_REQ_CREDIT_BUREAU_YEAR": 2,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_FAMILY_STATUS": "Married",
    "NAME_HOUSING_TYPE": "House / apartment",
    "OCCUPATION_TYPE": "Laborers",
    "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "FONDKAPREMONT_MODE": "reg oper account",
    "HOUSETYPE_MODE": "block of flats",
    "WALLSMATERIAL_MODE": "Panel",
    "EMERGENCYSTATE_MODE": "No"
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
- `MODEL_PATH`: Path to saved model file
- `DEBUG`: Enable debug mode
- `PORT`: Server port

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
