from app.services.s3 import S3
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
# from app.utils.response import api_json_response_format
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
import uuid
s3 = S3()

class BackgroundTask:
    def __init__(self, executor):        
        self.executor = executor
    
    
    async def train_model_background(self, base_path, user_name):
        
        json_file_name = "class_to_idx.json"

        base_path = f"{base_path.split('/')[0]}/"
        temp_base_path = base_path.split('/')[0]

        try: 
            class_names = await s3.get_dirs(base_path, "train/")

            # if not class_names:
            #     return api_json_response_format(False, "Class folder not found. Please create new class", 404, {})

            print(f"[*] class names -> {class_names}")

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

                    
                result = await s3.upload_file(base_path, file_like_obj, file_name=json_file_name)

                if result.get('success'):
                    os.remove(json_file_name)
                    print(f"Deleted local file: {json_file_name}")
                else:
                    print("Failed to upload json file to S3")
            
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
            
            
            with open(pth_file_path, "rb") as f:
                    data = f.read()
                    file_like_obj = io.BytesIO(data)
                    file_like_obj.seek(0) 

            result = await s3.upload_file(base_path, file_like_obj, file_name=pth_file_name )

            message = "PTH file uploaded successfully."

            if not result.get("success"):
                message = f"PTH file not uploaded. Error: {result.get('message')}"
                
            status = 'y'
            query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
            value = (status, temp_base_path, user_name)
            res = current_app.database.update_query(query,value)
            if res['data'] > 0:
                print("Value updated.")
            else:
                print(f"Error: str{res['message']}")

            print(message)
            # Clean up memory
            vars_to_delete = [
                "model", "train_loader", "val_loader", "train_dataset", "val_dataset",
                "class_names", "custom_class_to_idx", "file_like_obj", "data",
                "outputs", "preds", "labels"
            ]

            for var in vars_to_delete:
                if var in locals():
                    del locals()[var]

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("Model training completed")

        except Exception as e:
            print("Exception occured. Error : "+str(e))
        

    def train_model_async(self,base_path,user_name):
        task_id = uuid.uuid4().hex  
        self.executor.submit_stored(task_id, self.train_model_background, base_path, user_name)  
        print(task_id)   
        return task_id
