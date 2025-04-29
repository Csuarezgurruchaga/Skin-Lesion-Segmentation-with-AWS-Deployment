import json
import boto3
import base64
from requests_toolbelt.multipart import decoder
import logging
import os
import uuid

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize S3 client
s3 = boto3.client("s3")

# Configuration
BUCKET_NAME = 'skin-segmentation-app'
INPUT_PREFIX = 'input/'
ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def lambda_handler(event, context):
    """
    AWS Lambda handler function that processes a file upload request,
    decodes multipart/form-data, and saves the image to an S3 bucket.
    """
    logger.info("Receiving image upload request")
    logger.info(f"Event structure: {json.dumps(event, default=str)}")
    
    try:
        # Normalize headers to lowercase to avoid case sensitivity issues
        raw_headers = event.get("headers", {})
        headers = {k.lower(): v for k, v in raw_headers.items()}
        content_type = headers.get("content-type")
        
        if not content_type:
            logger.error("Missing 'Content-Type' header")
            return build_response(400, "Missing 'Content-Type' header")
        
        logger.info(f"Content-Type: {content_type}")
        
        # IMPORTANTE: Manejo mejorado de datos binarios
        body = event["body"]
        is_base64 = event.get("isBase64Encoded", False)
        logger.info(f"Body is base64 encoded: {is_base64}")
        
        if is_base64:
            try:
                logger.info(f"Base64 body length: {len(body) if body else 0}")
                body = base64.b64decode(body)
                logger.info(f"Decoded body length: {len(body) if body else 0}")
            except Exception as e:
                logger.error(f"Error decoding base64: {str(e)}")
                return build_response(400, f"Invalid base64 body: {str(e)}")
        elif isinstance(body, str):
            # No intentar codificar strings a bytes si no es necesario
            # Solo para multipart/form-data necesitamos convertir a bytes
            if "multipart/form-data" in content_type:
                logger.info("Converting string body to bytes for multipart decoding")
                body = body.encode("utf-8")
        
        # Verificación adicional
        if not body:
            logger.error("Empty request body")
            return build_response(400, "Empty request body")
            
        # Imprimir los primeros bytes para debug (sin exponer toda la data)
        if isinstance(body, bytes):
            logger.info(f"First 30 bytes of body: {body[:30]}")
        
        # Decode multipart/form-data
        try:
            logger.info("Attempting to decode multipart data")
            multipart_data = decoder.MultipartDecoder(body, content_type)
            logger.info(f"Multipart data decoded successfully with {len(multipart_data.parts)} parts")
        except Exception as e:
            logger.error(f"Error decoding multipart: {str(e)}")
            return build_response(400, f"Error decoding multipart/form-data: {str(e)}")
        
        # Process each part
        file_uploaded = False
        for i, part in enumerate(multipart_data.parts):
            logger.info(f"Processing part {i+1} of {len(multipart_data.parts)}")
            
            # Log part headers for debugging
            part_headers = {k.decode('utf-8') if isinstance(k, bytes) else k: 
                          v.decode('utf-8') if isinstance(v, bytes) else v 
                          for k, v in part.headers.items()}
            logger.info(f"Part headers: {part_headers}")
            
            content_disposition = part.headers.get(b"Content-Disposition", b"").decode()
            
            if "filename=" in content_disposition:
                # Extract filename
                filename = content_disposition.split("filename=")[1].strip('"')
                logger.info(f"Received file: {filename}")
                
                # Check file extension
                _, file_extension = os.path.splitext(filename.lower())
                if file_extension not in ALLOWED_EXTENSIONS:
                    logger.warning(f"File extension not allowed: {file_extension}")
                    return build_response(400, f"File type not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")
                
                # Generate unique key for S3
                unique_filename = f"{uuid.uuid4()}_{filename}"
                s3_key = f"{INPUT_PREFIX}{unique_filename}"
                
                # Get part content type
                part_content_type = part.headers.get(b"Content-Type", b"application/octet-stream").decode()
                logger.info(f"Part content type: {part_content_type}")
                
                # Verify we have content
                if not part.content or len(part.content) == 0:
                    logger.error("Empty file content received")
                    return build_response(400, "Empty file content received")
                
                logger.info(f"File content size: {len(part.content)} bytes")
                logger.info(f"First 30 bytes of file: {part.content[:30]}")
                
                try:
                    # Ensure correct content type for images
                    if part_content_type == "application/octet-stream":
                        if file_extension == '.jpg' or file_extension == '.jpeg':
                            part_content_type = "image/jpeg"
                        elif file_extension == '.png':
                            part_content_type = "image/png"
                    
                    # Upload file to S3
                    logger.info(f"Uploading to S3 with content type: {part_content_type}")
                    s3.put_object(
                        Bucket=BUCKET_NAME,
                        Key=s3_key,
                        Body=part.content,
                        ContentType=part_content_type
                    )
                    logger.info(f"File {filename} uploaded to s3://{BUCKET_NAME}/{s3_key}")
                    file_uploaded = True
                    
                    return build_response(200, {
                        "message": f"File '{filename}' uploaded successfully.",
                        "location": f"s3://{BUCKET_NAME}/{s3_key}",
                        "key": s3_key
                    })
                except Exception as e:
                    logger.error(f"Error uploading to S3: {str(e)}")
                    return build_response(500, f"Error uploading to S3: {str(e)}")
        
        if not file_uploaded:
            logger.warning("No file found in multipart data")
            return build_response(400, "No file found in the request")
    
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return build_response(500, f"Internal server error: {str(e)}")

def build_response(status_code, body):
    """Builds a standardized response for API Gateway"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # For CORS
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps(body) if isinstance(body, dict) else json.dumps({"message": body})
    }