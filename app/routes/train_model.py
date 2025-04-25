import os
import shutil
import random
import torch
import json
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from flask import Flask, request, Blueprint, current_app, jsonify
from app.utils.response import api_json_response_format
from app.services.authentication import Authentication
from app.services.s3 import S3
import copy
from functools import wraps
import io
from PIL import Image
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import time
import pandas as pd

load_dotenv()

s3 = S3()

train_bp = Blueprint("train", __name__)

BUCKET_NAME = os.getenv("BUCKET_NAME")

ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'])


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
        print("Class created successfully.")
        upload_result = await s3.upload_file(folder_name, images, image_class_name=image_class_name) 
        if upload_result.get('success'):
            print("Images uploaded successfully.")
            response = current_app.authentication.get_username(request)
            user_name = response.get('username')
            status = 'n'
            query = "INSERT INTO train_model (user_id, model_name, status)  SELECT user_id, %s, %s FROM users  WHERE user_name = %s;"
            value = (folder_name, status, user_name)
           
            res = current_app.database.execute_query(query,value)

            # if res["success"] and res["error_code"] == 200:
               
            #     else:
            #         print(f"Invalid Username...{username}")
            #         return api_json_response_format(False,str("Sorry, unable to authenticate. Invalid Username..."),401,{})

        else:
            print(f"Image not uploaded: {upload_result.get('message')}")
            return api_json_response_format(False, "Class created but image not uploaded.", 500, {}) 
        
        return api_json_response_format(True, "Class created successfully and Image uploaded.", 201, {})  
        
    else:
        print(f"Class {folder_name} already exist.")
        return api_json_response_format(False, result.get("message"), result.get("error_code"), {})
    
@train_bp.route('/list_dir', methods=['POST'])
@Authentication.token_required
async def list_dir():
    data = request.get_json()

    prefix = data.get('folder_path')

    dir_list = await s3.list_folder(request=request, prefix=prefix)

    if not dir_list:
        return api_json_response_format(False, "No folders", 400, {})
    
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

    if not result.get("success"):
        model_name = folder_path.split('/'[0])
        response = current_app.authentication.get_username(request)
        user_name = response.get('username')
        status = 'n'
        query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
        value = (status, model_name, user_name)
        
        res = current_app.database.update_query(query,value)
        if res['data'] > 0:
            print("Value updated.")
        else:
            print(f"Error: str{res['message']}")
        # if res["success"] and res["error_code"] == 200:
        #     if res["data"]:
        #         password = res["data"][0]["userpwd"]
        #     else:
        #         print(f"Invalid Username...{username}")
        #         return api_json_response_format(False,str("Sorry, unable to authenticate. Invalid Username..."),401,{})
        # return api_json_response_format(False, result.get('message'), result.get('error_code'), {})
    

    return api_json_response_format(True, "Image uploaded successfully.", 200, {})
    

@train_bp.route('/train_model', methods=['POST'])
@Authentication.token_required
async def train_model():
    data = request.get_json()
    base_path = data.get('base_path')
    json_file_name = "class_to_idx.json"

    if not base_path:
        return api_json_response_format(False, "Folder name not found", 404, {})
    
    # train_dirs = await s3.get_dirs(path, "train")
    # val_dirs = await s3.get_dirs(path, "val")

    # train_dir = f"{base_path}train"
    # val_dir = f"{base_path}val"
    base_path = f"{base_path.split('/')[0]}/"
    temp_base_path = base_path.split('/')[0]
    response = current_app.authentication.get_username(request)
    user_name = response.get('username')
    query = "SELECT tr.status FROM train_model AS tr LEFT JOIN users AS us ON tr.user_id = us.user_id WHERE us.user_name = %s AND tr.model_name = %s"
    value = (user_name, temp_base_path)
    res = current_app.database.execute_query(query,value)

    if res['data']:
        if res['data'][0]['status'] == 'y':
            return api_json_response_format(True, "Model already trained", 200, {})

    try: 
        class_names = await s3.get_dirs(base_path, "train/")

        if not class_names:
            return api_json_response_format(False, "Class folder not found. Please create new class", 404, {})

        class_json = await s3.get_dirs(base_path, "class_to_idx.json", "file")

        custom_class_to_idx = {name: i for i, name in enumerate(class_names)}

        if not class_json:
            
            with open(json_file_name, "w") as json_file:
                json.dump(custom_class_to_idx, json_file, indent=4)
                print("Saved class-to-ID mapping to 'class_to_idx.json'")

            with open(json_file_name, "rb") as f:
                data = f.read()
                file_like_obj = io.BytesIO(data)
                file_like_obj.seek(0) 

            result = await s3.upload_file(base_path, file_like_obj, file_name=json_file_name )

            if result.get('success'):
                os.remove(json_file_name)
                print(f"Deleted local file: {json_file_name}")
            else:
                return api_json_response_format(False, "Failed to upload json file to S3", 500, {})
        
        # tmp_dir = tempfile.mkdtemp()
        tmp_dir = 'temp'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        os.makedirs(tmp_dir, exist_ok=True)
        local_train_path = os.path.join(tmp_dir, f"{base_path}train/")
        local_val_path = os.path.join(tmp_dir, f"{base_path}val/")

        for split in ['train', 'val']:
            for class_name in class_names:
                temp_folder_path = f"{tmp_dir}/{base_path}{split}/{class_name.rsplit('/')[-2]}"
                os.makedirs(temp_folder_path, exist_ok=True)

        await s3.download_folder(f"{base_path}train/", local_train_path)
        await s3.download_folder(f"{base_path}val/", local_val_path)
        num_classes = len(custom_class_to_idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"train_dir -> {local_train_path}")
        train_dataset = datasets.ImageFolder(local_train_path, transform=train_transform)
        val_dataset = datasets.ImageFolder(local_val_path, transform=val_transform)
        train_dataset.class_to_idx = custom_class_to_idx
        val_dataset.class_to_idx = custom_class_to_idx

        class_names = sorted({os.path.basename(os.path.dirname(path)) for path, _ in train_dataset.samples})
        custom_class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        train_dataset.samples = [(path, custom_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in train_dataset.samples]
        val_dataset.samples = [(path, custom_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in val_dataset.samples]
        

        train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=5)

        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_acc = 0
        best_model = copy.deepcopy(model.state_dict())

        start_time = time.time()
        for epoch in range(100):
            model.train()
            train_loss, correct, total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            train_acc = correct / total * 100

            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu())
                    val_labels.extend(labels.cpu())

            val_acc = accuracy_score(val_labels, val_preds) * 100
            print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = copy.deepcopy(model.state_dict())
                print(">> New best model saved!")
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Elapsed time: {elapsed:.4f} seconds")

        
        pth_file_name = f"{temp_base_path}_graph_classifier.pth"
        pth_file_path = tmp_dir+"/"+pth_file_name
        torch.save(best_model, pth_file_path)

        print("Training completed. Best model saved.")
        
        status = 'y'
        query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
        value = (status, temp_base_path, user_name)
        res = current_app.database.update_query(query,value)
        if res['data'] > 0:
            print("Value updated.")
        else:
            print(f"Error: str{res['message']}")
        with open(pth_file_path, "rb") as f:
                data = f.read()
                file_like_obj = io.BytesIO(data)
                file_like_obj.seek(0) 

        result = await s3.upload_file(base_path, file_like_obj, file_name=pth_file_name )

        message = "PTH file uploaded successfully."

        if not result.get("success"):
            message = f"PTH file not uploaded. Error: {result.get('message')}"

        print(message)

        return api_json_response_format(True, "Training completed. Best model saved.", 200, {})



    except Exception as e:
        print("Exception occured. Error : "+str(e))
        return api_json_response_format(False, str(e), 500, {})


@train_bp.route('/start_predict', methods=['POST'])
@Authentication.token_required
async def start_predict():
    try:
        folder_name = request.form.get('folder_name')
        load_model = request.form.get('load_model')
        uploaded_files = request.files.getlist('images')
        excel_file = request.files.get('excel_file')

        if not folder_name:
            return api_json_response_format(False, "Folder not found.", 404, {})

        temp_folder_name = folder_name.split('/')[0]
        json_file_path = f"{temp_folder_name}/class_to_idx.json"
        pth_file_path = f"{temp_folder_name}/{temp_folder_name}_graph_classifier.pth"

        ext = excel_file.filename.split('.')[-1]
        try:
            if ext == "ods":
                excel = pd.read_excel(excel_file, engine="odf")
            elif ext in ["xls", "xlsx"]:
                excel = pd.read_excel(excel_file, engine="openpyxl")
            else:
                print("Unsupported file format.")
                return api_json_response_format(False, f"{excel_file.filename} is unsupported file format.", 400, {})
        except Exception as e:
            print(f"Error reading Excel file: {e}")

        
      
        if load_model:
            # Check if model files exist in S3
            is_folder_exist = await s3.get_dirs(folder_name, check=True)
            is_json_exist = await s3.get_dirs(json_file_path, type="file")
            is_pth_exist = await s3.get_dirs(pth_file_path, type="file")

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
                    print("All transition times are ≤ 60 ms.")
                    transition = "All transition times are ≤ 60 ms."
                else:
                    print("\nTransition times > 60 ms:")
                    if "Tap changer transition" in excel.columns:
                        transition_output = result[[transition_col, "Tap changer transition"]].to_string(index=False)
                    else:
                        transition_output = result[[transition_col]].to_string(index=False)

                    print(transition_output)
                    transition = f"{len(result)} rows with transition time > 60 ms."
            else:
                print("'Transition time' column not found.")
                transition = "'Transition time' column not found."

            results.append({
                "transition": transition,
                "output": transition_output
            })




            for filename in image_files:
                image_path = os.path.join(upload_path, filename)
                try:
                    pred_class, confidence = predict_image(image_path)
                    results.append({
                        "filename": filename,
                        "predicted_class": pred_class,
                        "confidence": confidence,
                        "corrected": False
                    })
                except Exception as e:
                    results.append({
                        "filename": filename,
                        "error": str(e)
                    })

            return api_json_response_format(True, "Prediction completed", 200, {"results": results})

        return api_json_response_format(False, "Invalid request", 400, {})

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
            return api_json_response_format(False, "Class name not found.", 400, {})
        
        if not user_response:
            return api_json_response_format(False, "Please select Yes or No", 400, {})
        
        if not images:
            # if os.path.exists(tmp_dir):
            #     with open('predict'+class_name, "rb") as file:
            #         data = file.read()
            #         image_file = io.BytesIO(data)
            #         image_file.seek(0) 

                # result = await s3.upload_file(base_path, file_like_obj, file_name=json_file_name )

            return api_json_response_format(False, "Image not found.", 400, {})

        model_name = class_name.split('/')[0]
        image_class_name = class_name.split('/')[-1]
        
        if user_response == "no" and class_name:
            response = await s3.upload_file(class_name, images, image_class_name=image_class_name)

        if response.get('success'):
            response = current_app.authentication.get_username(request)
            user_name = response.get('username')
            status = 'n'
            query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
            value = (status, model_name, user_name)
            
            res = current_app.database.update_query(query,value)
            if res['data'] > 0:
                print("Value updated.")
            else:
                print(f"Error: str{res['message']}")
            return api_json_response_format(True, f"Picture added to model {model_name}", 200, {})
        else:
            return api_json_response_format(False, response.get('message'), response.get('error_code'), {})


        

    except Exception as e:
        return api_json_response_format(False, f"Error: {str(e)}", 500, {})




