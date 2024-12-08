import numpy as np
import polars as pl
from typing import List, Dict, Union
from category_encoders import WOEEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Define feature groups as module-level constants
BINARY_FEATURES = [
    "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "FLAG_MOBIL", "FLAG_EMP_PHONE", 
    "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL", 
    "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION", 
    "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY", 
    "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", 
    "FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", 
    "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", "FLAG_DOCUMENT_7", 
    "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", 
    "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", 
    "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", 
    "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", 
    "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"
]

NUMERICAL_FEATURES = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", 
    "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", 
    "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", 
    "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", 
    "HOUR_APPR_PROCESS_START", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", 
    "APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", 
    "YEARS_BUILD_AVG", "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", 
    "FLOORSMAX_AVG", "FLOORSMIN_AVG", "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", 
    "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG", 
    "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE", 
    "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", 
    "FLOORSMAX_MODE", "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", 
    "LIVINGAREA_MODE", "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", 
    "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", "YEARS_BEGINEXPLUATATION_MEDI", 
    "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", "ENTRANCES_MEDI", 
    "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", 
    "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", 
    "TOTALAREA_MODE", "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", 
    "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE", 
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", 
    "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON", 
    "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"
]

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"
]

def preprocess_features(data: Dict[str, List[float]]) -> np.ndarray:
    """
    Preprocess input features to match model requirements.
    
    Args:
        data: Dictionary containing feature values
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    try:
        # Convert to Polars DataFrame
        df = pl.DataFrame(data)
        
        # Convert Y/N to binary for binary features
        for col in BINARY_FEATURES:
            if col in df.columns:
                df = df.with_columns(
                    pl.when(pl.col(col) == "Y").then(1)
                    .when(pl.col(col) == "N").then(0)
                    .otherwise(None)
                    .cast(pl.Int64)
                    .alias(col)
                )
        
        # Convert to pandas for WOE encoding
        df_pandas = df.to_pandas()
        
        # Load and apply pre-fitted WOE encoder
        encoder = joblib.load(Path('models/woe_encoder.joblib'))
        df_encoded = encoder.transform(df_pandas)
        
        # Convert back to polars
        df = pl.from_pandas(df_encoded)
        
        # Standardize numerical features
        # Note: If you have a saved scaler, load it here
        # scaler = joblib.load(Path('models/scaler.joblib'))
        scaler = StandardScaler()
        
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df = df.with_columns(
                    pl.Series(
                        name=col,
                        values=scaler.fit_transform(df[col].to_numpy().reshape(-1, 1)).flatten()
                    )
                )
        
        return df.to_numpy()
    
    except Exception as e:
        raise ValueError(f"Error preprocessing features: {str(e)}")

def format_prediction(prediction: np.ndarray) -> Dict[str, Union[float, str]]:
    """
    Format model prediction into API response.
    
    Args:
        prediction: Raw model prediction
        
    Returns:
        Formatted prediction with probability and risk category
    """
    prob = prediction[0][1]  # Probability of default
    
    # Risk categorization based on probability
    if prob < 0.2:
        risk_category = "Low Risk"
    elif prob < 0.4:
        risk_category = "Medium Risk"
    else:
        risk_category = "High Risk"
    
    return {
        "default_probability": float(prob),
        "risk_category": risk_category
    }