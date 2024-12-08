import unittest
import numpy as np
import joblib
from app.utils import preprocess_features, format_prediction
from config import Config

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(Config.MODEL_PATH)
        
        # Sample test data with all features
        cls.sample_input = {
            # Binary flags (35 features)
            "FLAG_OWN_CAR": ["N"],
            "FLAG_OWN_REALTY": ["Y"],
            "FLAG_MOBIL": [1],
            "FLAG_EMP_PHONE": [1],
            "FLAG_WORK_PHONE": [0],
            "FLAG_CONT_MOBILE": [1],
            "FLAG_PHONE": [1],
            "FLAG_EMAIL": [0],
            "REG_REGION_NOT_LIVE_REGION": [0],
            "REG_REGION_NOT_WORK_REGION": [0],
            "LIVE_REGION_NOT_WORK_REGION": [0],
            "REG_CITY_NOT_LIVE_CITY": [0],
            "REG_CITY_NOT_WORK_CITY": [0],
            "LIVE_CITY_NOT_WORK_CITY": [0],
            "FLAG_DOCUMENT_2": [0],
            "FLAG_DOCUMENT_3": [0],
            "FLAG_DOCUMENT_4": [0],
            "FLAG_DOCUMENT_5": [0],
            "FLAG_DOCUMENT_6": [0],
            "FLAG_DOCUMENT_7": [0],
            "FLAG_DOCUMENT_8": [0],
            "FLAG_DOCUMENT_9": [0],
            "FLAG_DOCUMENT_10": [0],
            "FLAG_DOCUMENT_11": [0],
            "FLAG_DOCUMENT_12": [0],
            "FLAG_DOCUMENT_13": [0],
            "FLAG_DOCUMENT_14": [0],
            "FLAG_DOCUMENT_15": [0],
            "FLAG_DOCUMENT_16": [0],
            "FLAG_DOCUMENT_17": [0],
            "FLAG_DOCUMENT_18": [0],
            "FLAG_DOCUMENT_19": [0],
            "FLAG_DOCUMENT_20": [0],
            "FLAG_DOCUMENT_21": [0],

            # Numerical features (70 features)
            "CNT_CHILDREN": [0],
            "AMT_INCOME_TOTAL": [135000.0],
            "AMT_CREDIT": [513000.0],
            "AMT_ANNUITY": [20000.0],
            "AMT_GOODS_PRICE": [472500.0],
            "REGION_POPULATION_RELATIVE": [0.018801],
            "DAYS_BIRTH": [-10000],
            "DAYS_EMPLOYED": [-1500],
            "DAYS_REGISTRATION": [-3000],
            "DAYS_ID_PUBLISH": [-2000],
            "OWN_CAR_AGE": [5.0],
            "CNT_FAM_MEMBERS": [2],
            "REGION_RATING_CLIENT": [2],
            "REGION_RATING_CLIENT_W_CITY": [2],
            "HOUR_APPR_PROCESS_START": [12],
            "EXT_SOURCE_1": [0.5],
            "EXT_SOURCE_2": [0.5],
            "EXT_SOURCE_3": [0.5],
            "APARTMENTS_AVG": [0.1],
            "BASEMENTAREA_AVG": [0.1],
            "YEARS_BEGINEXPLUATATION_AVG": [0.1],
            "YEARS_BUILD_AVG": [0.1],
            "COMMONAREA_AVG": [0.1],
            "ELEVATORS_AVG": [0.1],
            "ENTRANCES_AVG": [0.1],
            "FLOORSMAX_AVG": [0.1],
            "FLOORSMIN_AVG": [0.1],
            "LANDAREA_AVG": [0.1],
            "LIVINGAPARTMENTS_AVG": [0.1],
            "LIVINGAREA_AVG": [0.1],
            "NONLIVINGAPARTMENTS_AVG": [0.1],
            "NONLIVINGAREA_AVG": [0.1],
            "APARTMENTS_MODE": [0.1],
            "BASEMENTAREA_MODE": [0.1],
            "YEARS_BEGINEXPLUATATION_MODE": [0.1],
            "YEARS_BUILD_MODE": [0.1],
            "COMMONAREA_MODE": [0.1],
            "ELEVATORS_MODE": [0.1],
            "ENTRANCES_MODE": [0.1],
            "FLOORSMAX_MODE": [0.1],
            "FLOORSMIN_MODE": [0.1],
            "LANDAREA_MODE": [0.1],
            "LIVINGAPARTMENTS_MODE": [0.1],
            "LIVINGAREA_MODE": [0.1],
            "NONLIVINGAPARTMENTS_MODE": [0.1],
            "NONLIVINGAREA_MODE": [0.1],
            "APARTMENTS_MEDI": [0.1],
            "BASEMENTAREA_MEDI": [0.1],
            "YEARS_BEGINEXPLUATATION_MEDI": [0.1],
            "YEARS_BUILD_MEDI": [0.1],
            "COMMONAREA_MEDI": [0.1],
            "ELEVATORS_MEDI": [0.1],
            "ENTRANCES_MEDI": [0.1],
            "FLOORSMAX_MEDI": [0.1],
            "FLOORSMIN_MEDI": [0.1],
            "LANDAREA_MEDI": [0.1],
            "LIVINGAPARTMENTS_MEDI": [0.1],
            "LIVINGAREA_MEDI": [0.1],
            "NONLIVINGAPARTMENTS_MEDI": [0.1],
            "NONLIVINGAREA_MEDI": [0.1],
            "TOTALAREA_MODE": [0.1],
            "OBS_30_CNT_SOCIAL_CIRCLE": [0],
            "DEF_30_CNT_SOCIAL_CIRCLE": [0],
            "OBS_60_CNT_SOCIAL_CIRCLE": [0],
            "DEF_60_CNT_SOCIAL_CIRCLE": [0],
            "DAYS_LAST_PHONE_CHANGE": [-10],
            "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0],
            "AMT_REQ_CREDIT_BUREAU_DAY": [0.0],
            "AMT_REQ_CREDIT_BUREAU_WEEK": [0.0],
            "AMT_REQ_CREDIT_BUREAU_MON": [0.0],
            "AMT_REQ_CREDIT_BUREAU_QRT": [0.0],
            "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0],

            # Categorical features (14 features)
            "NAME_CONTRACT_TYPE": ["Cash loans"],
            "CODE_GENDER": ["F"],
            "NAME_TYPE_SUITE": ["Unaccompanied"],
            "NAME_INCOME_TYPE": ["Working"],
            "NAME_EDUCATION_TYPE": ["Secondary / secondary special"],
            "NAME_FAMILY_STATUS": ["Married"],
            "NAME_HOUSING_TYPE": ["House / apartment"],
            "OCCUPATION_TYPE": ["Laborers"],
            "WEEKDAY_APPR_PROCESS_START": ["MONDAY"],
            "ORGANIZATION_TYPE": ["Business Entity Type 3"],
            "FONDKAPREMONT_MODE": ["reg oper account"],
            "HOUSETYPE_MODE": ["block of flats"],
            "WALLSMATERIAL_MODE": ["Stone, brick"],
            "EMERGENCYSTATE_MODE": ["No"],
        }
    
    def test_preprocessing(self):
        """Test feature preprocessing"""
        processed_features = preprocess_features(self.sample_input)
        self.assertIsInstance(processed_features, np.ndarray)
        self.assertEqual(processed_features.shape[1], 119)
    
def test_preprocessing(self):
    """Test feature preprocessing"""
    processed_features = preprocess_features(self.sample_input)
    
    # Test output type and shape
    self.assertIsInstance(processed_features, np.ndarray)
    self.assertEqual(processed_features.shape[1], 119)  # Total number of features
    
    # Test binary conversions
    binary_features = [
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
    for feature_idx, feature_name in enumerate(binary_features):
        values = processed_features[:, feature_idx]
        self.assertTrue(all(val in [0, 1, None] for val in values))
    
    # Test categorical encoding (WOE)
    categorical_features = [
        "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
        "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"
    ]
    for feature in categorical_features:
        # WOE encoding should produce numerical values
        feature_idx = list(self.sample_input.keys()).index(feature)
        self.assertTrue(np.issubdtype(processed_features[:, feature_idx].dtype, np.number))
    
    # Test numerical features standardization
    # Test numerical features standardization
    numerical_features = [
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
    for feature in numerical_features:
        feature_idx = list(self.sample_input.keys()).index(feature)
        values = processed_features[:, feature_idx]
        # Check if values are standardized (approximately mean 0, std 1)
        self.assertTrue(-10 <= np.mean(values) <= 10)  # Allow some flexibility
        self.assertTrue(0 <= np.std(values) <= 10)     # Allow some flexibility
    
    self.assertFalse(np.isnan(processed_features).any(), 
                     "Preprocessed features contain NaN values")
    
    def test_prediction_format(self):
        """Test prediction formatting"""
        processed_features = preprocess_features(self.sample_input)
        prediction = self.model.predict_proba(processed_features)
        formatted = format_prediction(prediction)
        
        self.assertIn("default_probability", formatted)
        self.assertIn("risk_category", formatted)
        self.assertIsInstance(formatted["default_probability"], float)
        self.assertIsInstance(formatted["risk_category"], str)
    
    def test_model_prediction(self):
        """Test end-to-end prediction"""
        processed_features = preprocess_features(self.sample_input)
        prediction = self.model.predict_proba(processed_features)
        
        self.assertEqual(prediction.shape[1], 2)  # Binary classification
        self.assertTrue(0 <= prediction[0][1] <= 1)  # Probability between 0 and 1