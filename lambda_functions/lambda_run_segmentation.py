import os
import json
import tempfile
import boto3
import tensorflow as tf
import numpy as np
import cv2
import time
import logging
from inference_functions import preprocess_for_prediction, postprocess_prediction, overlay_segmentation, image_to_bytes

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 and CloudWatch clients
s3_client = boto3.client('s3')
cloudwatch_client = boto3.client('cloudwatch')

# Load the model (will be done when the container initializes)
MODEL_PATH = '/tmp/model/unet_skin_lesion_saved_model'
model = None

def download_and_extract_model(bucket, key, model_path):
    """Download and extract the model from S3."""
    try:
        # Create temporary directory if it doesn't exist
        if not os.path.exists('/tmp'):
            os.makedirs('/tmp')
        
        # Download model ZIP file from S3
        zip_path = '/tmp/model.zip'
        s3_client.download_file(bucket, key, zip_path)
        
        # Extract ZIP
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('/tmp')
        
        # Remove ZIP after extraction
        os.remove(zip_path)
        
        logger.info(f"Model successfully downloaded and extracted to {model_path}")
    except Exception as e:
        logger.error(f"Error downloading and extracting model: {str(e)}")
        raise e

def load_model():
    """Load the TensorFlow SavedModel."""
    global model
    try:
        model = tf.saved_model.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

def publish_metric(name, value, unit):
    """Publish a metric to CloudWatch."""
    try:
        cloudwatch_client.put_metric_data(
            Namespace='SkinLesionSegmentation',
            MetricData=[
                {
                    'MetricName': name,
                    'Value': value,
                    'Unit': unit
                }
            ]
        )
    except Exception as e:
        logger.error(f"Error publishing metric {name}: {str(e)}")

def lambda_handler(event, context):
    """Main function that runs when Lambda is triggered."""
    global model
    
    # Record processing start time
    start_time = time.time()
    
    try:
        # Verify we have an S3 event
        if 'Records' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps('No records found in event')
            }
        
        # Get bucket and object information
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Ignore non-image files
        if not key.lower().endswith(('.png', '.jpg', '.jpeg')):
            logger.info(f"Ignoring incompatible file: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps('File ignored: not a compatible image')
            }
        
        # Check if the file is in the output directory to avoid loops
        if '/output/' in key or key.startswith('output/'):
            logger.info(f"Ignoring file in output directory: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps('Ignoring file in output directory')
            }
        
        # Verify file is in input directory
        if not ('/input/' in key or key.startswith('input/')):
            logger.info(f"Ignoring file outside input directory: {key}")
            return {
                'statusCode': 200,
                'body': json.dumps('File ignored: not in input directory')
            }
        
        # Load the model if not loaded
        if model is None:
            # Download and extract model if it doesn't exist
            model_key = "model/unet_skin_lesion_saved_model.zip"
            if not os.path.exists(MODEL_PATH):
                download_and_extract_model(bucket, model_key, MODEL_PATH)
            
            # Load the model
            load_model()
        
        # Download image from S3
        logger.info(f"Downloading image from s3://{bucket}/{key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_bytes = response['Body'].read()
        
        # Preprocess the image
        img = preprocess_for_prediction(image_bytes)
        
        # Get prediction function
        prediction_function = model.signatures['serving_default']
        
        # Get input tensor name
        input_tensor_name = prediction_function.inputs[0].name
        input_tensor_name_parts = input_tensor_name.split(':')
        tensor_name = input_tensor_name_parts[0].split('_')[-1]
        
        # Create input dictionary for the model
        inputs = {tensor_name: tf.convert_to_tensor(img)}
        
        # Perform prediction
        prediction_start = time.time()
        prediction_dict = prediction_function(**inputs)
        prediction_time = time.time() - prediction_start
        
        # Extract prediction tensor
        output_key = list(prediction_dict.keys())[0]
        prediction_tensor = prediction_dict[output_key]
        prediction_np = prediction_tensor.numpy()
        
        # Process prediction to get binary mask
        mask = postprocess_prediction(prediction_np)
        
        # Convert original image to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Create segmentation overlay
        overlay = overlay_segmentation(original_img, mask)
        
        # Convert result to bytes
        result_bytes = image_to_bytes(overlay)
        
        # Generate output filename
        filename = os.path.basename(key)
        base_name, ext = os.path.splitext(filename)
        
        # Ensure output path is in the output directory
        output_key = f"output/segmented_{base_name}{ext}"
        
        # Save result to S3
        s3_client.put_object(
            Bucket="skin-segmentation-app-output",
            Key=output_key,
            Body=result_bytes,
            ContentType=f'image/{ext.lstrip(".")}' if ext.lstrip(".") != "jpg" else "image/jpeg"
        )
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Publish metrics to CloudWatch
        publish_metric('ProcessingTime', total_time, 'Seconds')
        publish_metric('PredictionTime', prediction_time, 'Seconds')
        
        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Segmented image saved to s3://{bucket}/{output_key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'bucket': bucket,
                'input_key': key,
                'output_key': output_key,
                'processing_time': total_time,
                'prediction_time': prediction_time,
            })
        }
    
    except Exception as e:
        # Log error
        logger.error(f"Error in lambda_handler: {str(e)}")
        
        # Publish error metric
        publish_metric('ProcessingErrors', 1, 'Count')
        
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }