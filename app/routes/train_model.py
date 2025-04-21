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


load_dotenv()

s3 = S3()

train_bp = Blueprint("train", __name__)

BUCKET_NAME = os.getenv("BUCKET_NAME")

ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg'])


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
        else:
            print(f"Image not uploaded: {result.get('message')}")
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

    dir_list = await s3.list_folder(prefix)

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
        return api_json_response_format(False, result.get('message'), result.get('error_code'), {})
    

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
        

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        best_acc = 0
        best_model = copy.deepcopy(model.state_dict())

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
        temp_base_path = base_path.split('/')[0]
        pth_file_name = f"{temp_base_path}_graph_classifier.pth"
        pth_file_path = tmp_dir+"/"+pth_file_name
        torch.save(best_model, pth_file_path)

        print("Training completed. Best model saved.")
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

        
# @train_bp.route('/start_predict', methods=['POST'])
# @Authentication.token_required
# async def start_predict():

#     data = request.get_json()

#     folder_name = data.get('folder_name')
#     temp_folder_name = folder_name.split('/')[0]
#     load_model = data.get('load_model')
#     # folder_name = f"{folder_name.split('/')[0]}/"

#     if not folder_name:
#         return api_json_response_format(False, "Folder name not found.", 404, {})
    
#     json_file_path = f"{temp_folder_name}/class_to_idx.json"

#     pth_file_path = f"{temp_folder_name}/{temp_folder_name}_graph_classifier.pth"
#     if load_model:
#         is_folder_exist = await s3.get_dirs(folder_name, check=True)
#         is_json_exist = await s3.get_dirs(json_file_path, type="file")
#         is_pth_exist = await s3.get_dirs(pth_file_path, type="file")

#         if not is_folder_exist:
#             return api_json_response_format(False, "Folder path not found.", 404, {})
        
#         tmp_dir = 'predict'
#         if os.path.exists(tmp_dir):
#             shutil.rmtree(tmp_dir)

#         os.makedirs(tmp_dir, exist_ok=True)

#         if not is_json_exist:
#             return api_json_response_format(False, "Class to idx json file found. Please train your model", 404, {})

#         if not is_pth_exist:
#             return api_json_response_format(False, "PTH file not found. Please train your model.", 404, {})
        
#         await s3.download_folder(temp_folder_name, tmp_dir+"/"+temp_folder_name)
#         print(f"{temp_folder_name} class files downloaded from S3.")
        
#         local_json_path = tmp_dir+"/"+json_file_path
#         local_pth_path = tmp_dir+"/"+pth_file_path
#         local_image_path = tmp_dir+"/"+folder_name

#         with open(local_json_path, 'r') as f:
#             class_to_idx = json.load(f)
#         idx_to_class = {v: k for k, v in class_to_idx.items()}
#         class_names = [idx_to_class[i] for i in sorted(idx_to_class)]
        

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         model.fc = nn.Linear(model.fc.in_features, len(class_names))
#         model.load_state_dict(torch.load(local_pth_path, map_location=device))
#         model = model.to(device)
#         model.eval()

#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                 [0.229, 0.224, 0.225])
#         ])

#     def predict_image(image_path):
#         image = Image.open(image_path).convert("RGB")
#         input_tensor = transform(image).unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = model(input_tensor)
#             probabilities = torch.softmax(output, dim=1)
#             confidence, predicted = torch.max(probabilities, 1)
#         return class_names[predicted.item()], confidence.item() * 100
    
#     supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
#     image_files = [f for f in os.listdir(local_image_path) if f.lower().endswith(supported_exts)]

#     if not image_files:
#         print("No valid image files found.")
#         return api_json_response_format(False, "No valid image files found.", 404, {})
    
#     for filename in image_files:
#         image_path = os.path.join(local_image_path, filename)
#         try:
#             pred_class, confidence = predict_image(image_path)
#             print(f"\n{filename} -> Predicted: {pred_class} ({confidence:.2f}%)")
#             user_input = input("Is this correct? (y/n): ").strip().lower()

#             if user_input == "exit":
#                 break
#             elif user_input == "y":
#                 continue  # do nothing
#             elif user_input == "n":
#                 print("Available classes:")
#                 for i, cname in enumerate(class_names):
#                     print(f"{i + 1}. {cname}")
#                 try:
#                     correct_index = int(input("Enter correct class number: ")) - 1
#                     correct_class = class_names[correct_index]
#                 except:
#                     print("Invalid input. Skipping this image.")
#                     continue

#                 # Copy the image to both train and val folders of the correct class
#                 dest_train = os.path.join(base_path, "train", correct_class)
#                 dest_val = os.path.join(base_path, "val", correct_class)
#                 os.makedirs(dest_train, exist_ok=True)
#                 os.makedirs(dest_val, exist_ok=True)

#                 shutil.copy(image_path, os.path.join(dest_train, filename))
#                 shutil.copy(image_path, os.path.join(dest_val, filename))
#                 print(f"Copied to: {dest_train} and {dest_val}")
#         except Exception as e:
#             print(f" Error processing {filename}: {e}")

    
    
#     response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f"{folder_name}{folder_name}_graph_classifier.pth", MaxKeys=1)
#     if not response.get('KeyCount', 0) > 0:
#         return api_json_response_format(False, "Modle file not found. Please train the model first.", 404, {})



@train_bp.route('/start_predict', methods=['POST'])
@Authentication.token_required
async def start_predict():
    try:
        data = request.get_json()
        folder_name = data.get('folder_name')
        load_model = data.get('load_model', False)

        if not folder_name:
            return api_json_response_format(False, "Folder name not found.", 404, {})

        temp_folder_name = folder_name.split('/')[0]
        json_file_path = f"{temp_folder_name}/class_to_idx.json"
        pth_file_path = f"{temp_folder_name}/{temp_folder_name}_graph_classifier.pth"

        if load_model:
            # Check if files exist in S3
            is_folder_exist = await s3.get_dirs(folder_name, check=True)
            is_json_exist = await s3.get_dirs(json_file_path, type="file")
            is_pth_exist = await s3.get_dirs(pth_file_path, type="file")

            if not is_folder_exist:
                return api_json_response_format(False, "Folder path not found.", 404, {})
            if not is_json_exist:
                return api_json_response_format(False, "Class to idx JSON file not found.", 404, {})
            if not is_pth_exist:
                return api_json_response_format(False, "PTH model file not found.", 404, {})

            # Setup temp directories
            tmp_dir = 'predict'
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

            # Download from S3
            await s3.download_folder(temp_folder_name, os.path.join(tmp_dir, temp_folder_name))

            local_json_path = os.path.join(tmp_dir, json_file_path)
            local_pth_path = os.path.join(tmp_dir, pth_file_path)
            local_image_path = os.path.join(tmp_dir, folder_name)
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

            # Process all valid images
            supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
            image_files = [f for f in os.listdir(local_image_path) if f.lower().endswith(supported_exts)]

            if not image_files:
                return api_json_response_format(False, "No valid image files found.", 404, {})

            results = []
            for filename in image_files:
                image_path = os.path.join(local_image_path, filename)
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
        data = request.get_json()
        corrections = data.get("corrections", [])
        folder_name = data.get("folder_name")

        if not folder_name or not corrections:
            return api_json_response_format(False, "Missing folder_name or corrections.", 400, {})

        temp_folder_name = folder_name.split('/')[0]
        tmp_dir = 'predict'
        local_image_path = os.path.join(tmp_dir, folder_name)
        base_path = os.path.join(tmp_dir, temp_folder_name)

        results = []
        for correction in corrections:
            filename = correction.get("filename")
            user_response = correction.get("response")  # "yes" or "no"
            corrected_class = correction.get("corrected_class")  # Only needed if "no"

            if not filename or not user_response:
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "reason": "Missing filename or response"
                })
                continue

            image_path = os.path.join(local_image_path, filename)
            if not os.path.exists(image_path):
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "reason": "Image file not found"
                })
                continue

            if user_response == "yes":
                # No action needed if prediction is correct
                results.append({
                    "filename": filename,
                    "status": "correct"
                })
            elif user_response == "no" and corrected_class:
                # Add the image to the corrected class folder
                dest_train = os.path.join(base_path, "train", corrected_class)
                dest_val = os.path.join(base_path, "val", corrected_class)
                os.makedirs(dest_train, exist_ok=True)
                os.makedirs(dest_val, exist_ok=True)

                shutil.copy(image_path, os.path.join(dest_train, filename))
                shutil.copy(image_path, os.path.join(dest_val, filename))

                results.append({
                    "filename": filename,
                    "corrected_class": corrected_class,
                    "status": "copied"
                })
            else:
                results.append({
                    "filename": filename,
                    "status": "skipped",
                    "reason": "Invalid response or corrected_class missing"
                })

        return api_json_response_format(True, "Corrections processed", 200, {"results": results})

    except Exception as e:
        return api_json_response_format(False, f"Server error: {str(e)}", 500, {})



    




    
