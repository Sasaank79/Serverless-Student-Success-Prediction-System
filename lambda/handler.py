import json
import logging
from src.preprocessing import DataPipeline
from src.model import ModelLoader

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize globally for warm starts
pipeline = DataPipeline()
# model = ModelLoader("src/model.pkl") # Uncomment when artifact exists

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    """
    try:
        body = json.loads(event.get('body', '{}'))
        logger.info(f"Received event: {body}")

        # Inference logic
        # processed_data = pipeline.transform(body)
        # prediction = model.predict(processed_data)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "prediction": "dummy_prediction"
            })
        }
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
