import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
# S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
# AWS_REGION = os.environ.get("AWS_REGION")

BUCKET_NAME = os.getenv("BUCKET_NAME")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
AWS_REGION = os.environ.get("AWS_REGION")

class AWS_S3:        
 
    def get_s3_client(self):
        try:                                          
            s3_client__obj = boto3.client('s3', aws_access_key_id=S3_ACCESS_KEY, aws_secret_access_key=S3_SECRET_KEY,region_name=AWS_REGION)             
            return s3_client__obj
        except Exception as error:
            print("[X] error in get_s3_client(): ",str(error), flush=True)


    

   
