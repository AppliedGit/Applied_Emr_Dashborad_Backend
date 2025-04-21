import os
from dotenv import load_dotenv
from .aws_resource import AWS_S3
from app.utils.response import api_json_response_format
from concurrent.futures import ThreadPoolExecutor
import asyncio
from io import BytesIO


executor = ThreadPoolExecutor()

aws_s3 = AWS_S3()

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg'])
s3 = aws_s3.get_s3_client()
class S3:
    
    async def create_folders(self, base_folder, class_names, splits=['train', 'val']):
        try:
            base_folder = base_folder.strip("/")

            # Check if base folder already has any objects
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{base_folder}/", MaxKeys=1)
            if response.get('KeyCount', 0) > 0:
                return api_json_response_format(False, f"Folder '{base_folder}' already exists.", 400, {})

            # Always include a top-level folder marker
            s3.put_object(Bucket=BUCKET_NAME, Key=f"{base_folder}/")

            # Create each subfolder
            for split in splits:
                split_path = f"{base_folder}/{split}/"
                s3.put_object(Bucket=BUCKET_NAME, Key=split_path)

                for class_name in class_names:
                    class_folder_path = f"{split_path}{class_name}/"
                    s3.put_object(Bucket=BUCKET_NAME, Key=class_folder_path)

            return api_json_response_format(True, "Folders created successfully.", 201, {})
        
        except Exception as e:
            return api_json_response_format(False, str(e), 500, {})


    
    async def list_folder(self, prefix=""):
        files = []

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        all_keys = []
        for page in pages:
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if key != prefix:  # skip the folder itself
                    all_keys.append(key)

        # Build tree from keys
        return self.build_folder_tree(all_keys)

    def build_folder_tree(self, s3_keys):
        root = {"name": "root", "type": "folder", "children": []}

        # Normalize keys and remove empty ones
        # s3_keys = [key.strip("/") for key in s3_keys if key.strip()]
        all_prefixes = set()

        # Build a set of all folder-like prefixes
        for key in s3_keys:
            parts = key.split("/")
            for i in range(1, len(parts)):
                all_prefixes.add("/".join(parts[:i]))

        def insert_path(path_parts, current_node, current_path):
            if not path_parts:
                return

            part = path_parts[0]
            if part == "":
                insert_path(path_parts[1:], current_node, current_path)
                return

            next_path = f"{current_path}/{part}" if current_path else part
            is_folder = next_path in all_prefixes
            node_type = "folder" if is_folder else "file"

            for child in current_node["children"]:
                if child["name"] == part and child["type"] == node_type:
                    insert_path(path_parts[1:], child, next_path)
                    return

            new_node = {
                "name": part,
                "type": node_type,
                "path": next_path + ("/" if is_folder else "")
            }


            if node_type == "folder":
                new_node["children"] = []
                insert_path(path_parts[1:], new_node, next_path)

            current_node["children"].append(new_node)

        for key in s3_keys:
            parts = key.split("/")
            insert_path(parts, root, "")

        return root["children"]
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS
    async def upload_file(self, folder_name, files, file_name="", image_class_name=""):
        try:
            if image_class_name:
                folder_name = f"{folder_name.split('/')[0]}"
                image_class_name = f"{image_class_name.split('/')[0]}"
                for file in files:
                    if self.allowed_file(file.filename):
                        file_content = file.read()
                        for split in ['train', 'val']:
                            file_path = f"{folder_name}/{split}/{image_class_name}/{file.filename}"
                            s3.upload_fileobj(BytesIO(file_content), BUCKET_NAME, file_path)
                    else:
                        return api_json_response_format(False, "Image format not supported.", 400, {})
            elif file_name:
                file_path = f"{folder_name}{file_name}"
                s3.upload_fileobj(files, BUCKET_NAME, file_path)
            else:
                
                temp_folder = f"{folder_name.split('/')[0]}"
                class_name = f"{folder_name.rsplit('/')[-1]}"
                for file in files:
                    file_content = file.read()
                    for split in ['train', 'val']:
                        file_path = f"{temp_folder}/{split}/{class_name}/{file.filename}"
                        s3.upload_fileobj(BytesIO(file_content), BUCKET_NAME, file_path)       
            
            return api_json_response_format (True,"Image uploaded successfully", 200, {})
        except Exception as e:
            print(f"Error uploading file to S3: {e}")
            return api_json_response_format(False, str(e), 500, {})

    
    async def get_dirs(self, path, folder="", type="",check=False):

        dirs = []
        prefix = f"{path}{folder}"
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{path}{folder}")

        if response.get('KeyCount', 0) > 0:
            
            for obj in response.get('Contents'):
                key = obj.get('Key')
                if key.endswith('/') and check:
                    dirs.append(key) 
                if key.endswith('/') and key != prefix:
                    dirs.append(key)
                if not key.endswith('/') and type == 'file':
                    dirs.append(key)
        print(dirs)

        return dirs
    
    async def download_file_async(self, s3_key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f'Downloading {s3_key} to {local_path}')
        await asyncio.get_event_loop().run_in_executor(
            executor, s3.download_file, BUCKET_NAME, s3_key, local_path
        )

    async def download_folder(self, s3_folder_path, local_folder_path):
        paginator = s3.get_paginator('list_objects_v2')
        download_tasks = []

        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder_path):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                if s3_key.endswith('/'):
                    # Skip directory-like keys
                    continue

                relative_path = os.path.relpath(s3_key, s3_folder_path)
                if relative_path in ('.', '..'):
                    # Avoid accidental overwrite
                    continue

                local_path = os.path.join(local_folder_path, relative_path)
                task = self.download_file_async(s3_key, local_path)
                download_tasks.append(task)

        await asyncio.gather(*download_tasks)

                    

        



