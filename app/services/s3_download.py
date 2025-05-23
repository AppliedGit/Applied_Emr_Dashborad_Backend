import boto3
import botocore
import os
from app.services.emr_logger import write_iiot_log
from flask import request, current_app

import threading
import signal
import sys
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION")


# def graceful_shutdown(signum, frame):
#     print("Shutting down gracefully...")
#     sys.exit(0)

# signal.signal(signal.SIGTERM, graceful_shutdown)

def download_s3_files(s3_folder_path, local_folder_path):
    try:
        # Create session INSIDE the function
        session = boto3.session.Session()
        s3 = session.client(
            's3',
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name=AWS_REGION
        )

        paginator = s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder_path):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                write_iiot_log(0, s3_key)
                
                if s3_key.endswith('/') or s3_key.endswith('classifier.pth'):
                    continue

                relative_path = os.path.relpath(s3_key, s3_folder_path)
                if relative_path in ('.', '..'):
                    continue

                local_path = os.path.join(local_folder_path, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                write_iiot_log(0, f"{s3_key}  local path : {local_path}")
                write_iiot_log(0, "download file name for train : " + s3_key)

                s3.download_file(BUCKET_NAME, s3_key, local_path)

    except Exception as error:
        write_iiot_log(0, "Error in download_s3_files " + str(error))




# def download_s3_files(bucket_name, local_dir):
#     s3 = boto3.client('s3')
#     paginator = s3.get_paginator('list_objects_v2')

#     for page in paginator.paginate(Bucket=BUCKET_NAME):
#         for obj in page.get('Contents', []):
#             key = obj['Key']
#             local_path = os.path.join(local_dir, key)
#             os.makedirs(os.path.dirname(local_path), exist_ok=True)
#             s3.download_file(bucket_name, key, local_path)
#             print(f"Downloaded: {key} to {local_path}")

class S3_test:    

    def download_folder_prediction_test(self, s3_folder_path, local_folder_path, user_name=None, progress_callback=None):
        try: 
            s3 = boto3.client(    's3',    aws_access_key_id=S3_ACCESS_KEY,    aws_secret_access_key=S3_SECRET_KEY,region_name=AWS_REGION)
            paginator = s3.get_paginator('list_objects_v2')

            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder_path):
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    write_iiot_log(0,s3_key)
                    if s3_key.endswith('/') or s3_key.endswith('classifier.pth'):
                        continue

                    relative_path = os.path.relpath(s3_key, s3_folder_path)
                    if relative_path in ('.', '..'):
                        continue

                    local_path = os.path.join(local_folder_path, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    write_iiot_log(0,s3_key+"  local path : "+local_path)
                    write_iiot_log(0,"download file name for train : "+s3_key)           
                    s3.download_file(BUCKET_NAME, s3_key, local_path)
            
        except Exception as error:
            # print(error)
            write_iiot_log(0," Exception in download_folder_prediction():  "+str(error))


    # def start_download_in_thread(self, s3_folder_path, local_folder_path, user_name=None, progress_callback=None):
    #     thread = threading.Thread(
    #         target=self.download_folder_prediction_test,
    #         args=(s3_folder_path, local_folder_path, user_name, progress_callback)
    #     )
    #     thread.start()
    #     return thread

    