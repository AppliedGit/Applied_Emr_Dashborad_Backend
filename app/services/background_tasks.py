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
import asyncio
from app.services.emr_logger import write_iiot_log

s3 = S3()

class BackgroundTask:
    def __init__(self, executor):        
        self.executor = executor
        self.user_progress = {}

    def clear_progress(self, user_name):
        if user_name in self.user_progress:
            del self.user_progress[user_name]
            print("[*] user progress deleted.", flush=True)
            

    async def send_progress_update(self, user_name, epoch=0, train_acc=0, val_acc=0, message=''):
        # Update progress dictionary using user_id
        self.user_progress[user_name] = {
            'epoch': epoch,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'message': message
        }

    # This function retrieves the current progress for a user_id
    def get_progress(self, user_name):
        progress = self.user_progress.get(user_name)
        if progress:
            return json.dumps(progress)
        return None
    
    async def update_model_status(self,model_id,epoch,train_acc,val_acc,message=''):
        try:
            # print(model_id,model_id,epoch,train_acc,val_acc,message)
            query = "insert into  model_status (model_id,epoch,train_acc,val_acc,message) values(%s,%s,%s,%s,%s) "
            value = (model_id,epoch,train_acc,val_acc,message)
            res = current_app.database.update_query(query,value)
            # print(res)
            if res['data'] > 0:
                pass
                # print("[*] Value updated.", flush=True)
            else:
                print(f"[X] Error: {res['message']}", flush=True)
        except Exception as error:
            write_iiot_log(1,"Error in update_model_status() : "+str(error))
            print("Error in update_model_status() : "+str(error))
    
    async def train_model_background(self, base_path, user_name,model_id):
        
        json_file_name = "class_to_idx.json"

        print("base_path  ",base_path)
        print("user_name  ",user_name)
        

        base_path = f"{base_path.split('/')[0]}/"
        temp_base_path = base_path.split('/')[0]
        write_iiot_log(0,"temp_base_path  "+temp_base_path)
        

        try:
            current_app.database.connect()
            
            query = "delete from  model_status  "
            value = ()
            res = current_app.database.update_query(query,value)
            print(res)
            if res['data'] > 0:
                print("[*] Value updated.", flush=True)
            else:
                print(f"[X] Error: {res['message']}", flush=True)
            # self.update_model_status(model_id,'','','','Training process initiated...')
        except Exception as error:
            print("Error in delete model  : "+str(error))

        try: 
            write_iiot_log(0,"before s3 get directory  "+base_path)
            class_names = await s3.get_dirs(f"{base_path}train/")

            write_iiot_log(0,"after s3 get directory class_names  ")

            # if not class_names:
            #     return api_json_response_format(False, "Class folder not found. Please create new class", 404, {})

            print(f"[*] class names -> {class_names}", flush=True)
            write_iiot_log(0,"images files download process started  ")
            await self.update_model_status(model_id,'','','','images files download process started...')
            class_json = await s3.get_dirs(f"{base_path}{json_file_name}", file=True)
            write_iiot_log(0,"after files download process end  ")


            custom_class_to_idx = {name: i for i, name in enumerate(class_names)}

            if not class_json:
                
                with open(json_file_name, "w") as json_file:
                    json.dump(custom_class_to_idx, json_file, indent=4)
                    print("[*] Saved class-to-ID mapping to 'class_to_idx.json'", flush=True)

                with open(json_file_name, "rb") as f:
                    data = f.read()
                    file_like_obj = io.BytesIO(data)
                    file_like_obj.seek(0) 

                    
                result = await s3.upload_file(base_path, file_like_obj, file_name=json_file_name)

                if result.get('success'):
                    os.remove(json_file_name)
                    print(f"[*] Deleted local file: {json_file_name}", flush=True)
                else:
                    print("[X] Failed to upload json file to S3", flush=True)
            
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

            # await self.send_progress_update(user_name, message="Downloading training and validation data...")
            await self.update_model_status(model_id,'','','','Downloading training and validation data...')
            await asyncio.sleep(1)
            start_time = time.time()

            await s3.download_folder(f"{base_path}train/", local_train_path, user_name=user_name, progress_callback=self.send_progress_update)
            await s3.download_folder(f"{base_path}val/", local_val_path, user_name=user_name, progress_callback=self.send_progress_update)
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
            print(f"[*] train_dir -> {local_train_path}", flush=True)
            await self.update_model_status(model_id,'','','','Training started...')
            # await self.send_progress_update(user_name, message="Training started...")
            
            train_dataset = datasets.ImageFolder(local_train_path, transform=train_transform)
            val_dataset = datasets.ImageFolder(local_val_path, transform=val_transform)
            train_dataset.class_to_idx = custom_class_to_idx
            val_dataset.class_to_idx = custom_class_to_idx

            class_names = sorted({os.path.basename(os.path.dirname(path)) for path, _ in train_dataset.samples})
            custom_class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

            train_dataset.samples = [(path, custom_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in train_dataset.samples]
            val_dataset.samples = [(path, custom_class_to_idx[os.path.basename(os.path.dirname(path))]) for path, _ in val_dataset.samples]
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

            # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            # val_loader = DataLoader(val_dataset, batch_size=32)

            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model = model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_acc = 0
            best_model = copy.deepcopy(model.state_dict())

            epoch = 0
            val_acc = 0
            train_acc = 0
            print("training started ......................")
            
            for epoch in range(50):
                write_iiot_log(0,epoch)
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
                train_acc = round(correct / total * 100, 2)

                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)
                        _, preds = torch.max(outputs, 1)
                        val_preds.extend(preds.cpu())
                        val_labels.extend(labels.cpu())

                val_acc =  round(accuracy_score(val_labels, val_preds) * 100,2)
                # print(f"[*] Epoch {epoch+1}: Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%", flush=True)
                write_iiot_log(0,"epoch : "+str(epoch))
                await self.update_model_status(model_id,epoch,train_acc,val_acc)
                # await self.send_progress_update(user_name, epoch+1, train_acc, val_acc)

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(model.state_dict())
                    write_iiot_log(0,"[*] >> New best model saved!")
            
            pth_file_name = f"{temp_base_path}_graph_classifier.pth"
            # pth_file_path = tmp_dir+"/"+pth_file_name
            pth_file_path = "Img_models/"+pth_file_name
            
            print("model path : --------- ",pth_file_path)
            torch.save(best_model, pth_file_path)
            write_iiot_log(0,"model saved successfully --------- start upload process  "+pth_file_path)
            
            # with open(pth_file_path, "rb") as f:
            #         data = f.read()
            #         file_like_obj = io.BytesIO(data)
            #         file_like_obj.seek(0) 

            # result = await s3.upload_file(base_path, file_like_obj, file_name=pth_file_name )

            

            # if not result.get("success"):
            #     message = f"PTH file not uploaded. Error: {result.get('message')}"
            # else:
            #     message = "PTH file uploaded successfully."
            #     self.update_model_status(model_id,epoch+1, train_acc, val_acc, message=f"PTH file created successfully.")
            message = "PTH file uploaded successfully."
            write_iiot_log(0,message)
            self.update_model_status(model_id,epoch+1, train_acc, val_acc, message=f"PTH file created successfully.")
                # await self.send_progress_update(user_name, epoch+1, train_acc, val_acc, message=f"PTH file created successfully.")
                
            status = 'trained'
            # query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
            # value = (status, temp_base_path, user_name)

            query = "UPDATE train_model  SET status = %s  WHERE model_id = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
            write_iiot_log(0,status+" "+str(model_id)+"  "+user_name)
            value = (status, str(model_id), user_name)


            print(query,value)
            res = current_app.database.update_query(query,value)
            if res['data'] > 0:
                print("[*] Value updated.", flush=True)
            else:
                print(f"[X] Error: {res['message']}", flush=True)

            end_time = time.time()
            elapsed = end_time - start_time
            write_iiot_log(0,f"[*] Elapsed time: {elapsed:.4f} seconds")
            completed_time = round(elapsed / 60, 2)            
            await self.update_model_status(model_id,epoch+1, train_acc, val_acc, message=f"Training completed in {completed_time} minutes.")
            time.sleep(2)
            # await self.send_progress_update(user_name, epoch+1, train_acc, val_acc, message=f"Training completed in {completed_time} minutes.")

            
            write_iiot_log(0,message)
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
            write_iiot_log(0,"[*] Model training completed")

            # with open("/shared/training_done", "w") as f:
            #     f.write("done")
            # print("[*] Training done file created.", flush=True)

        except Exception as e:
            write_iiot_log(1,"exception occured. error "+str(e))
            print("[X] Exception occured. Error : "+str(e), flush=True)
        

    # def train_model_async(self,base_path,user_name):
    #     task_id = uuid.uuid4().hex  
    #     self.executor.submit_stored(task_id, self.train_model_background, base_path, user_name)  
    #     print(task_id)   
    #     return task_id
    
    def train_model_async(self, base_path, user_name,model_id):
        task_id = uuid.uuid4().hex

        def run_training():
            asyncio.run(self.train_model_background(base_path, user_name,model_id))

        self.executor.submit_stored(task_id, run_training)        
        return task_id

