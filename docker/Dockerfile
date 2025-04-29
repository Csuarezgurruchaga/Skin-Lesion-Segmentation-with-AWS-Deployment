FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

RUN yum update -y && \
    yum install -y \
    gcc \
    g++ \
    make \
    cmake \
    unzip \
    wget \
    libgl1-mesa-glx

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt

COPY lambda_run_segmentation.py ${LAMBDA_TASK_ROOT}

COPY inference_functions.py ${LAMBDA_TASK_ROOT}

CMD ["lambda_run_segmentation.lambda_handler"]
