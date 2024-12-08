import unittest
import json
from app.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        
    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
    def test_prediction(self):
        test_data = {
            'features': [[1.0, 2.0, 3.0]]
        }
        response = self.app.post(
            '/predict',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)