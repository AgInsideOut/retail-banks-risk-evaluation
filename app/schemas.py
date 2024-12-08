from marshmallow import Schema, fields, ValidationError

class PredictionSchema(Schema):
    features = fields.List(fields.List(fields.Float()), required=True)