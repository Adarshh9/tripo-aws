import os
from utils import process_image, run_model
from boto3 import Session
import torch
import pickle
import datetime
import gzip

session = Session(profile_name='AlphaIntern24')
s3 = session.client('s3')

def load_model():
    with gzip.open('model_quantized_compressed.pkl.gz', 'rb') as f_in:
        model_data = f_in.read()

    model = pickle.loads(model_data)

    print("Model Loaded")
    return model

def fetch_image_from_s3(image_path, output_dir):
    bucket_name = image_path.split('/')[2]
    key = image_path.split('/')[-1]
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    
    # Save the image locally in the output directory
    os.makedirs(output_dir, exist_ok=True)
    local_image_path = os.path.join(output_dir, 'input.png')
    with open(local_image_path, 'wb') as f:
        f.write(obj['Body'].read())
    
    return local_image_path

def upload_to_s3(file_path, bucket_name, s3_key):
    with open(file_path, 'rb') as f:
        s3.upload_fileobj(f, bucket_name, s3_key)
    s3_url = f's3://{bucket_name}/{s3_key}'
    return s3_url

def generate_mesh(image_path, output_dir):
    print('inside generate_mesh')
    print('Process start')

    try:
        # Fetch the image from S3
        local_image_path = fetch_image_from_s3(image_path, output_dir)
        print(f'Image fetched from S3: {local_image_path}')
    except Exception as e:
        print(f'Error fetching image from S3: {e}')
        return None

    try:
        # Process the image
        image = process_image(local_image_path, output_dir)
        print('Process end')
    except Exception as e:
        print(f'Error processing image: {e}')
        return None

    try:
        # Load the model
        print('Run start')
        model = load_model()
    except Exception as e:
        print(f'Error loading model: {e}')
        return None

    try:
        # Run the model
        output_file_path = run_model(model, image, output_dir)
        print('Run end')
    except Exception as e:
        print(f'Error running model: {e}')
        return None

    try:
        # Upload the generated mesh file to S3
        bucket_name = 'vasana-bkt1'
        s3_key = f'meshes/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}-{os.path.basename(output_file_path)}'
        s3_url = upload_to_s3(output_file_path, bucket_name, s3_key)
        print(f'File uploaded to S3: {s3_url}')
        return s3_url
    except Exception as e:
        print(f'Error uploading to S3: {e}')
        return None
