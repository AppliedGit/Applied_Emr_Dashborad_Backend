import os
from dotenv import load_dotenv
from .aws_resource import AWS_S3
from app.utils.response import api_json_response_format
from concurrent.futures import ThreadPoolExecutor
import asyncio
from io import BytesIO
from flask import request, current_app



executor = ThreadPoolExecutor()

aws_s3 = AWS_S3()

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'])
s3 = aws_s3.get_s3_client()
class S3:
    
    async def create_folders(self, base_folder, class_names, splits=['train', 'val']):
        try:
            base_folder = base_folder.strip("/")

            # Check if base folder already has any objects
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{base_folder}/", MaxKeys=1)
            if response.get('KeyCount', 0) > 0:
                return api_json_response_format(False, f"Folder '{base_folder}' already exists.", 400, {})

            
            s3.put_object(Bucket=BUCKET_NAME, Key=f"{base_folder}/")

            # Create subfolder
            for split in splits:
                split_path = f"{base_folder}/{split}/"
                s3.put_object(Bucket=BUCKET_NAME, Key=split_path)

                for class_name in class_names:
                    class_folder_path = f"{split_path}{class_name}/"
                    s3.put_object(Bucket=BUCKET_NAME, Key=class_folder_path)

            return api_json_response_format(True, "Folders created successfully.", 201, {})
        
        except Exception as e:
            return api_json_response_format(False, str(e), 500, {})
        
    # async def delete_s3_object(self, path):
    #     try:
    #         # First, check if the path exists exactly
    #         response = await asyncio.to_thread(
    #             s3.list_objects_v2,
    #             Bucket=BUCKET_NAME,
    #             Prefix=path
    #         )

    #         if 'Contents' not in response:
    #             return "No such file or folder"

    #         # If the path is exactly an object (single image)
    #         exact_match = any(obj['Key'] == path for obj in response['Contents'])

    #         if exact_match:
    #             # It is a file (image), delete it directly
    #             delete_response = await asyncio.to_thread(
    #                 s3.delete_object,
    #                 Bucket=BUCKET_NAME,
    #                 Key=path
    #             )
    #             if delete_response['ResponseMetadata']['HTTPStatusCode'] == 204 or delete_response['ResponseMetadata']['HTTPStatusCode'] == 200:
    #                 return True
    #             else:
    #                 return "Delete failed"
    #         else:
    #             # It is a folder, delete all inside
    #             # Ensure the path ends with '/'
    #             prefix = path if path.endswith('/') else path + '/'
    #             response = await asyncio.to_thread(
    #                 s3.list_objects_v2,
    #                 Bucket=BUCKET_NAME,
    #                 Prefix=prefix
    #             )
    #             if 'Contents' not in response:
    #                 return "No such folder"

    #             delete_requests = [{'Key': obj['Key']} for obj in response['Contents']]

    #             delete_response = await asyncio.to_thread(
    #                 s3.delete_objects,
    #                 Bucket=BUCKET_NAME,
    #                 Delete={'Objects': delete_requests}
    #             )
    #             if delete_response['ResponseMetadata']['HTTPStatusCode'] == 200:
    #                 return True
    #             else:
    #                 return "Delete failed"
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return str(e)
    async def delete_s3_object(self, path):
        try:
            prefix = path if path.endswith('/') else path + '/'

            # List all objects under the prefix (folder or file)
            response = await asyncio.to_thread(
                s3.list_objects_v2,
                Bucket=BUCKET_NAME,
                Prefix=prefix
            )

            # If nothing found under the prefix
            if 'Contents' not in response:
                # Check if it's a single file instead of a folder
                file_response = await asyncio.to_thread(
                    s3.list_objects_v2,
                    Bucket=BUCKET_NAME,
                    Prefix=path
                )

                # If no file either, return error
                if 'Contents' not in file_response:
                    return "No such file or folder"

                exact_match = any(obj['Key'] == path for obj in file_response['Contents'])

                if exact_match:
                    # It's a file (not folder), delete directly
                    delete_response = await asyncio.to_thread(
                        s3.delete_object,
                        Bucket=BUCKET_NAME,
                        Key=path
                    )
                    if delete_response['ResponseMetadata']['HTTPStatusCode'] in [200, 204]:
                        await self._delete_counterpart(path)
                        return True
                    else:
                        return "Delete failed"
                else:
                    return "No such file or folder"

            # Delete all objects under the folder
            delete_requests = [{'Key': obj['Key']} for obj in response['Contents']]

            # Try to delete the folder marker if it exists
            if not path.endswith('/'):
                delete_requests.append({'Key': path + '/'})
            else:
                delete_requests.append({'Key': path})

            delete_response = await asyncio.to_thread(
                s3.delete_objects,
                Bucket=BUCKET_NAME,
                Delete={'Objects': delete_requests}
            )

            if delete_response['ResponseMetadata']['HTTPStatusCode'] == 200:
                await self._delete_counterpart(prefix)
                return True
            else:
                return "Delete failed"

        except Exception as e:
            print(f"[X] Error: {e}")
            return str(e)



    # Helper method to delete the counterpart path (train <-> val)
    async def _delete_counterpart(self, original_path):
        try:
            if "train/" in original_path:
                counterpart = original_path.replace("train/", "val/")
            elif "val/" in original_path:
                counterpart = original_path.replace("val/", "train/")
            else:
                return

            # Recursively delete the counterpart
            await self.delete_s3_object(counterpart)
        except Exception as e:
            print(f"[X] Error deleting counterpart: {e}")


    
    async def list_folder(self, request, prefix="", ):
        files = []

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix)

        all_keys = []
        for page in pages:
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if key != prefix:  
                    all_keys.append(key)

        root = self.build_folder_tree(all_keys)

        
        response = current_app.authentication.get_username(request)
        if isinstance(response, bytes):
            response = response.decode('utf-8')
        if isinstance(response, str):
            import json
            response = json.loads(response)
        user_name = response.get("username")

       
        await self.add_model_status_to_folders(root, user_name)

        return root["children"]


    def build_folder_tree(self, s3_keys):
        root = {"name": "root", "type": "folder", "children": []}
        all_prefixes = set()
        cdn_base_url = "https://d3dmqth8jvbx2a.cloudfront.net"

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

            
            if node_type == "file":
                new_node["cdn_url"] = f"{cdn_base_url}/{next_path}"

            if node_type == "folder":
                new_node["children"] = []
                insert_path(path_parts[1:], new_node, next_path)

            current_node["children"].append(new_node)

        for key in s3_keys:
            parts = key.split("/")
            insert_path(parts, root, "")

        return root


    async def add_model_status_to_folders(self, root, user_name):
        for node in root.get("children", []):
            if node["type"] == "folder":
                model_name = node["name"]
                query = """
                    SELECT status FROM train_model 
                    WHERE model_name = %s AND user_id = (
                        SELECT user_id FROM users WHERE user_name = %s
                    )
                """
                value = (model_name, 'admin')
                result = current_app.database.execute_query(query, value)

                if result["data"]:
                    node["status"] = result["data"][0]["status"] if result.get('data') else "train"
                else:
                    node["status"] = 'train'



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
            print(f"[X] Error uploading file to S3: {e}")
            return api_json_response_format(False, str(e), 500, {})

    
    async def get_dirs(self, path, file=False ):

        result = []
        # prefix = f"{path}{folder}"

        # response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{path}{folder}")
        if not file:
            response = s3.list_objects_v2(
                Bucket=BUCKET_NAME,
                Prefix=path,
                Delimiter="/"
            )
            result = [p['Prefix'] for p in response.get('CommonPrefixes', [])]
            print(result) 
        else:
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=path)
            

            for obj in response.get("Contents", []):
                key = obj["Key"]
                if not key.endswith("/"):  # Ignore folders
                    result.append(key)
                    
        return result
    
    async def download_file_async(self, s3_key, local_path, user_name=None, progress_callback=None):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f'[*] Downloading {s3_key} to {local_path}')
        if progress_callback:
            await progress_callback(user_name, message=f"Downloading {s3_key}")
            await asyncio.sleep(1)
        await asyncio.get_event_loop().run_in_executor(
            executor, s3.download_file, BUCKET_NAME, s3_key, local_path
        )

    async def download_folder(self, s3_folder_path, local_folder_path, user_name=None, progress_callback=None):
        paginator = s3.get_paginator('list_objects_v2')

        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_folder_path):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                if s3_key.endswith('/'):
                    continue

                relative_path = os.path.relpath(s3_key, s3_folder_path)
                if relative_path in ('.', '..'):
                    continue

                local_path = os.path.join(local_folder_path, relative_path)
                await self.download_file_async(s3_key, local_path, user_name, progress_callback)
                await asyncio.sleep(1)  # ✅ delay between downloads

    async def check_empty_class_folders(self, base_folder):
        try:
            base_folder = base_folder.strip("/")
            train_path = f"{base_folder}/train/"

            # List all class folders under train/
            response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=train_path, Delimiter='/')

            if 'CommonPrefixes' not in response:
                return api_json_response_format(False, "No class folders found inside train/", 404, {})

            empty_folders = []

            for prefix_info in response['CommonPrefixes']:
                class_folder_path = prefix_info['Prefix']

                # Now check if any real files exist under this class folder
                class_files_response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=class_folder_path)

                # Check if there are files other than the folder itself
                file_count = 0
                for obj in class_files_response.get('Contents', []):
                    if not obj['Key'].endswith('/'):  # Ignore folder keys
                        file_count += 1

                if file_count == 0:
                    empty_folders.append(class_folder_path)

            if empty_folders:
                return api_json_response_format(False, "Empty folders found.", 404, {"empty_folders": empty_folders})
            else:
                return api_json_response_format(True, "No empty folders found.", 200, {})

        except Exception as e:
            return api_json_response_format(False, str(e), 500, {})
                        

            



