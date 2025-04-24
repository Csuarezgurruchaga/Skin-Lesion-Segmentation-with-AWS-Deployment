# Skin-Lesion-Segmentation-with-AWS-Deployment
This repository demonstrates a complete end-to-end pipeline for skin lesion (mole) segmentation using a U-Net model trained on the ISIC2018 dataset. The trained model is deployed to AWS using S3, API Gateway, Lambda functions, and ECR.

Key Features

Model Training: U-Net (depth=5) trained on ISIC2018 dataset, saved in TensorFlow SavedModel format.

Cloud Storage: Model artifacts stored in S3 under skin-segmentation-app/model.

Web Interface: app.py provides a UI to upload images via API Gateway.

Lambda Upload: lambda_upload.py parses JSON POST requests containing Base64‑encoded image data, decodes and validates the image, then stores it under the input/ prefix in S3.

Repository Structure:
