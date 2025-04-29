import os, io
import shutil
import torch
import time
import json
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from flask import  request, Blueprint, current_app, jsonify, stream_with_context, Response
from app.utils.response import api_json_response_format
from app.services.authentication import Authentication
from app.services.s3 import S3
from PIL import Image
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import pandas as pd

load_dotenv()

s3 = S3()

train_bp = Blueprint("train", __name__)

BUCKET_NAME = os.getenv("BUCKET_NAME")

ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'])

import logging



@train_bp.route('/create_class', methods=['POST'])
@Authentication.token_required
async def create_class():
    # data = request.get_json()

    # folder_name = data.get("folder_name")
    # no_of_classes = data.get("no_of_classes")
    # class_names = data.get("class_names")
    folder_name = request.form.get('folder_name')
    no_of_classes = request.form.get('no_of_classes')
    class_names_string = request.form.get('class_names')
    class_names = class_names_string.split(',')
    image_class_name = request.form.get('image_class_name')
    images = request.files.getlist('images')

    if not folder_name:
        return api_json_response_format(False, "Please provide folder name for your model.", 404, {})

    if not no_of_classes:
        return api_json_response_format(False, "Please provide number of classes.", 404, {})
    
    if not class_names:
        return api_json_response_format(False, "Please provide class names.", 404, {})
    
    if not image_class_name:
        return api_json_response_format(False, "Please provide class name where to upload.", 404, {})
    
    if not images:
        return api_json_response_format(False, "Please provide image.", 404, {})
    

    result = await s3.create_folders(folder_name, class_names)
    if result.get("success"):
        print("[*] Class created successfully.", flush=True)
        upload_result = await s3.upload_file(folder_name, images, image_class_name=image_class_name) 
        if upload_result.get('success'):
            print("[*] Images uploaded successfully.", flush=True)
            response = current_app.authentication.get_username(request)
            user_name = response.get('username')
            status = 'train'
            query = "INSERT INTO train_model (user_id, model_name, status)  SELECT user_id, %s, %s FROM users  WHERE user_name = %s;"
            value = (folder_name, status, user_name)
           
            res = current_app.database.execute_query(query,value)   
            print(res)

        else:
            print(f"[X] Image not uploaded: {upload_result.get('message')}", flush=True)
            return api_json_response_format(False, "Class created but image not uploaded.", 500, {}) 
        
        return api_json_response_format(True, "Class created successfully and Image uploaded.", 201, {})  
        
    else:
        print(f"[*] Class {folder_name} already exist.", flush=True)
        return api_json_response_format(False, result.get("message"), result.get("error_code"), {})
    
@train_bp.route('/list_dir', methods=['POST'])
@Authentication.token_required
async def list_dir():
    data = request.get_json()

    prefix = data.get('folder_path')

    dir_list = await s3.list_folder(request=request, prefix=prefix)

    if not dir_list:
        return api_json_response_format(False, "No folders", 404, {})
    
    return api_json_response_format(True, "Folder list", 200, dir_list)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@train_bp.route('/upload_image', methods=['POST'])
@Authentication.token_required
async def upload_image():
    folder_path = request.form.get('folder_path')
    files = request.files.getlist('files')

    if not folder_path:
        return api_json_response_format(False, "Folder path not found.", 404, {})
        
    result = await s3.upload_file(folder_path, files)

    if result.get("success"):
        model_name = folder_path.split('/')[0]
        response = current_app.authentication.get_username(request)
        user_name = response.get('username')
        json_file_name = "class_to_idx.json"
        response = await s3.delete_s3_object(f"{model_name}/{json_file_name}")
        if response == True:
                print(f"[*] Json file deleted successfully. Please train your model.", flush=True)
        status = 'train'
        query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
        value = (status, model_name, 'admin')
        res = current_app.database.update_query(query,value)
        if  res['data'] > 0:
            print("[*] Value updated.", flush=True)
        else:
            print(f"[X] Error: {res['message']}", flush=True)

    return api_json_response_format(True, "Image uploaded successfully.", 200, {})


@train_bp.route('/train_model', methods=['POST'])
@Authentication.token_required
async def train():

    data = request.get_json()
    base_path = data.get('base_path')

    if not base_path:
        return api_json_response_format(False, "Model name not found", 404, {})
    try:
        empty_folder = await s3.check_empty_class_folders(base_path)
        if not empty_folder['success']:
            return api_json_response_format(False, empty_folder['message'], empty_folder['error_code'], {})

        response = current_app.authentication.get_username(request)
        temp_base_path = base_path.split('/')[0]
        user_name = response.get('username')
        query = "SELECT tr.status FROM train_model AS tr LEFT JOIN users AS us ON tr.user_id = us.user_id WHERE us.user_name = %s AND tr.model_name = %s"
        value = (user_name, temp_base_path)
        res = current_app.database.execute_query(query,value)

        if res['data']:
            if res['data'][0]['status'] == 'trained':

                print("[*] Model already trained", flush=True)
                return api_json_response_format(True, "Model already trained", 200, {})

        current_app.background_runner.executor.submit(
            current_app.background_runner.train_model_async, base_path, user_name
        )
        status = 'training'
        query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
        value = (status, temp_base_path, user_name)
        res = current_app.database.update_query(query,value)
        if res['data'] > 0:
            print("[*] Value updated.", flush=True)
        else:
            print(f"[X] Error: str{res['message']}", flush=True)

        return api_json_response_format(True, "Model training started...", 200, {})

    except Exception as e:
        return api_json_response_format(False, f"Error: {str(e)}", 500, {})

@train_bp.route('/train_model_progress', methods=['GET'])

def get_train_model_progress():
    def generate():
        user_name = 'admin'
        last_message = None
        while True:
            
            progress = current_app.background_runner.get_progress(user_name)  
            if progress and progress != last_message:
                last_message = progress
                data_dict = json.loads(progress)
                yield f"data: {data_dict}\n\n"

                if  "Training completed" in data_dict['message']:
                    print(f"[*] Epoch 100 reached for user {user_name}, stopping stream.", flush=True)
                    current_app.background_runner.clear_progress(user_name)
                    break
            time.sleep(1)

    return Response(stream_with_context(generate()), content_type='text/event-stream')

@train_bp.route("/delete", methods=['POST'])
@Authentication.token_required
async def delete():
    # Initialize a session using Amazon S3
    data = request.get_json()
    path = data.get('path')
    try:
        # Delete the object
        response = await s3.delete_s3_object(path)
        if response == True:
            print(f"[*] {path} deleted successfully. Please train your model.", flush=True)

            response = current_app.authentication.get_username(request)
            temp_path = path.split('/')
            model_name = path.split('/')[0]
            user_name = response.get('username')
            
            if len(temp_path) > 2:
                json_file_name = "class_to_idx.json"
                response = await s3.delete_s3_object(f"{model_name}/{json_file_name}")
                if response == True:
                     print(f"[*] Json file deleted successfully. Please train your model.", flush=True)
                status = 'train'
                query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
                value = (status, model_name, user_name)
                res = current_app.database.update_query(query,value)

            else: 

                query = "DELETE FROM train_model WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
                value = (model_name, user_name)
                res = current_app.database.update_query(query,value)

            if res['data'] > 0:
                print("[*] Database value updated.",flush=True)
            else:
                print(f"[X] Error: str{res['message']}",flush=True)

            return api_json_response_format(True, "Object deleted successfully.", 200, {}) 
        else:
            return api_json_response_format(False, f"ERROR: {str(response)}", 500, {})

    except Exception as e:
        print(f"[X] Error: {e}", flush=True)
        return api_json_response_format(False, f"Error: {e}", 500, {})
    
@train_bp.route('/start_predict', methods=['POST'])
@Authentication.token_required
async def start_predict():
    try:
        folder_name = request.form.get('folder_name')
        # load_model = request.form.get('load_model')
        uploaded_files = request.files.getlist('images')
        excel_file = request.files.get('excel_file')

        if not folder_name:
            return api_json_response_format(False, "Folder not found.", 404, {})
        if not uploaded_files:
            return api_json_response_format(False, "Images not found.", 404, {})
        if not excel_file:
            return api_json_response_format(False, "CSV file not found.", 404, {})
        json_file_name = "class_to_idx.json"
        temp_folder_name = folder_name.split('/')[0]
        json_file_path = f"{temp_folder_name}/{json_file_name}"
        pth_file_path = f"{temp_folder_name}/{temp_folder_name}_graph_classifier.pth"

        ext = excel_file.filename.split('.')[-1]
        try:
            if ext == "ods":
                excel = pd.read_excel(excel_file, engine="odf")
            elif ext in ["xls", "xlsx"]:
                excel = pd.read_excel(excel_file, engine="openpyxl")
            else:
                print("[X] Unsupported file format.", flush=True)
                return api_json_response_format(False, f"{excel_file.filename} is unsupported file format.", 400, {})
        except Exception as e:
            print(f"[X] Error reading Excel file: {e}" ,flush=True)

        
      
       
        # Check if model files exist in S3
        is_folder_exist = await s3.get_dirs(folder_name)
        is_json_exist = await s3.get_dirs(json_file_path, file=True)
        is_pth_exist = await s3.get_dirs(pth_file_path, file=True)

        if not is_folder_exist:
            return api_json_response_format(False, "Folder path not found.", 404, {})
        if not is_json_exist:
            return api_json_response_format(False, "Class to idx JSON file not found.", 404, {})
        if not is_pth_exist:
            return api_json_response_format(False, "PTH model file not found.", 404, {})

        # Setup temp directory
        tmp_dir = 'predict'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # Save uploaded files to temp folder
        upload_path = os.path.join(tmp_dir, "uploaded_images")
        os.makedirs(upload_path, exist_ok=True)
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for file in uploaded_files:
            if file.filename.lower().endswith(supported_exts):
                filename = secure_filename(file.filename)
                file.save(os.path.join(upload_path, filename))

        # Download model + json from S3
        await s3.download_folder(temp_folder_name, os.path.join(tmp_dir, temp_folder_name))

        print("[*] Download completed", flush=True)

        local_json_path = os.path.join(tmp_dir, json_file_path)
        local_pth_path = os.path.join(tmp_dir, pth_file_path)
        base_path = os.path.join(tmp_dir, temp_folder_name)

        # Load class mappings
        with open(local_json_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load(local_pth_path, map_location=device))
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])

        def predict_image(image_path):
            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            return class_names[predicted.item()], confidence.item() * 100

        # Predict uploaded images
        image_files = [f for f in os.listdir(upload_path) if f.lower().endswith(supported_exts)]
        if not image_files:
            return api_json_response_format(False, "No valid image files found in uploaded data.", 404, {})

        results = []
        transition = None
        transition_output = None  

        transition_col = next((col for col in excel.columns if "transition time" in str(col).lower()), None)

        if transition_col:
            result = excel[excel[transition_col] > 60]
            if result.empty:
                print("All transition times are ≤ 60 ms.", flush=True)
                transition = "All transition times are ≤ 60 ms."
            else:
                print("\nTransition times > 60 ms:", flush=True)
                if "Tap changer transition" in excel.columns:
                    transition_output = result[[transition_col, "Tap changer transition"]].to_string(index=False)
                else:
                    transition_output = result[[transition_col]].to_string(index=False)

                print(transition_output, flush=True)
                transition = f"{len(result)} rows with transition time > 60 ms."
        else:
            print("[X] 'Transition time' column not found.", flush=True)
            transition = "'Transition time' column not found."


        for filename in image_files:
            image_path = os.path.join(upload_path, filename)
            try:
                pred_class, confidence = predict_image(image_path)
                results.append({
                    "filename": filename,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "corrected": False,
                    "transition": transition,
                    "output": transition_output
                })
            except Exception as e:
                results.append({
                    "filename": filename,
                    "error": str(e)
                })

        return api_json_response_format(True, "Prediction completed", 200, {"results": results})

    except Exception as e:
        return api_json_response_format(False, f"Server error: {str(e)}", 500, {})




@train_bp.route('/correct_predictions', methods=['POST'])
@Authentication.token_required
async def correct_predictions():
    try:
        
        class_name = request.form.get('class_name')
        user_response = request.form.get('user_response')
        images = request.files.getlist('images')

        if not class_name:
            return api_json_response_format(False, "Class name not found.", 404, {})
        
        if not user_response:
            return api_json_response_format(False, "Please select Yes or No", 404, {})
        
        if not images:
            # if os.path.exists(tmp_dir):
            #     with open('predict'+class_name, "rb") as file:
            #         data = file.read()
            #         image_file = io.BytesIO(data)
            #         image_file.seek(0) 

                # result = await s3.upload_file(base_path, file_like_obj, file_name=json_file_name )

            return api_json_response_format(False, "Image not found.", 404, {})

        model_name = class_name.split('/')[0]
        image_class_name = class_name.split('/')[-1]
        
        if user_response == "no" and class_name:
            response = await s3.upload_file(class_name, images, image_class_name=image_class_name)

        if response.get('success'):
            response = current_app.authentication.get_username(request)
            user_name = response.get('username')
            status = 'train'
            query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
            value = (status, model_name, 'admin')
            
            res = current_app.database.update_query(query,value)
            if res['data'] > 0:
                print("[*] Value updated.", flush=True)
            else:
                print(f"[X] Error: str{res['message']}", flush=True)
            return api_json_response_format(True, f"Picture added to model {model_name}", 200, {})
        else:
            return api_json_response_format(False, response.get('message'), response.get('error_code'), {})


        

    except Exception as e:
        return api_json_response_format(False, f"Error: {str(e)}", 500, {})

@train_bp.route('/task_status', methods=['POST'])
@Authentication.token_required
def get_task_status():
    data = request.get_json()
    task_id = data.get('task_id')
    future = current_app.background_runner.executor.futures.result(task_id)
    if not future:
        return jsonify({"status": "unknown", "message": "Task ID not found"}), 404

    if future.done():
        try:
            result = future.result()
            return jsonify({"status": "done", "result": result})
        except Exception as e:
            return jsonify({"status": "failed", "error": str(e)})
    return jsonify({"status": "running"})


