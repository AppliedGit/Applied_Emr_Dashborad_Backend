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
from app.services.emr_logger import write_iiot_log
import app.services.database as db_con
from app.services.s3_download import download_s3_files


load_dotenv()

s3 = S3()
# dd = S3_test()

train_bp = Blueprint("train", __name__)

BUCKET_NAME = os.getenv("BUCKET_NAME")
CDN_URL = "https://dd5uxxzzce6o3.cloudfront.net"

ALLOWED_EXTENSIONS = set(['xls', 'csv', 'png', 'jpeg', 'jpg', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp'])

import logging



@train_bp.route('/create_class', methods=['POST'])
@Authentication.token_required
async def create_class():
    # data = request.get_json()

    # folder_name = data.get("folder_name")
    # no_of_classes = data.get("no_of_classes")
    # class_names = data.get("class_names")


    # folder_name = "modal_one"
    # class_names = ["one"]
    # images_list = ["ABCN_Down_phase2.bmp"]

    folder_name = request.form.get('folder_name')
    no_of_classes = request.form.get('no_of_classes')
    class_names_string = request.form.get('class_names')
    class_names = class_names_string.split(',')
    image_class_name = request.form.get('image_class_name')
    images = request.files.getlist('images')
    images_list = images

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
        else:            
            return api_json_response_format(False, "Class created but image not uploaded.", 500, {}) 
        
        try:
            model_id,image_class_id = -1,-1
            query = "SELECT model_id FROM train_model  where model_name = %s "
            value = (folder_name,)
            res = current_app.database.execute_query(query,value)        
            if "data" in res:
                if len(res["data"]) > 0:   
                    model_id = res["data"][0]["model_id"]                    
                    query = "INSERT INTO train_class_name (model_id, class_name) values(%s,%s)"            
                    for sub_class_name in class_names:                                
                        value = (model_id, sub_class_name)                        
                        res = current_app.database.insert_query(query,value)  
                        if image_class_name == sub_class_name:
                            image_class_id = res["data"]                    
                    query = "INSERT INTO train_class_images (model_id, class_id,image_name) values(%s,%s,%s)"            
                    for image_name in images_list:                                                        
                        value = (model_id, image_class_id,image_name.filename)                        
                        res = current_app.database.insert_query(query,value)                                                             
            else:
                pass                            
        except Exception as error:
            write_iiot_log(1,"error in create_class database process "+str(error))
            
        
        return api_json_response_format(True, "Class created successfully and Image uploaded.", 201, {})  
        
    else:
        print(f"[*] Class {folder_name} already exist.", flush=True)
        return api_json_response_format(False, result.get("message"), result.get("error_code"), {})
    
@train_bp.route('/list_dir', methods=['POST'])
@Authentication.token_required
async def list_dir():
    try:
        data = request.get_json()
        prefix = data.get('folder_path')
        dir_list,dir_key_list = [],[]
        train_val_list = ["train","val"]        
        response = current_app.authentication.get_username(request)
        user_name = response.get('username')        
        if prefix == "":
            query = "SELECT model_id,model_name FROM train_model order by model_name "
            value = ()
            res = current_app.database.execute_query(query,value)
        else:
            if "/train" in prefix:
                prefix = prefix.replace("/train", "")
            elif "/val" in prefix:
                prefix = prefix.replace("/val", "")
            query = "SELECT model_id,model_name FROM train_model  where model_name = %s "
            value = (prefix,)
            res = current_app.database.execute_query(query,value)                    
        model_name_list,model_id_list = [],[]        
        if "data" in res:            
            if len(res["data"]) > 0:            
                for res_model_name in res["data"]:
                    model_name_list.append(res_model_name["model_name"])
                    model_id_list.append(res_model_name["model_id"])

        for index,model_name in enumerate(model_name_list):            
            dir_key_list.append(model_name+"/")            
            query = "SELECT id,model_id,class_name FROM train_class_name  where model_id = %s "
            value = (str(model_id_list[index]),)
            res = current_app.database.execute_query(query,value)            
            if "data" in res:                
                if len(res["data"]) > 0:
                    for train_val in train_val_list:                        
                        dir_key_list.append(model_name+"/"+train_val+"/")
                        for train_class_val in res["data"]:
                            sub_class_name  = train_class_val["class_name"]
                            sub_class_id = train_class_val["id"]                            
                            dir_key_list.append(model_name+"/"+train_val+"/"+sub_class_name+"/")
                            sub_query = "SELECT image_name FROM train_class_images  where model_id = %s  and class_id = %s"
                            sub_value = (str(model_id_list[index]),sub_class_id)
                            sub_res = current_app.database.execute_query(sub_query,sub_value)
                            if "data" in sub_res:
                                if len(sub_res["data"]) > 0:
                                    for sub_res_data  in sub_res["data"]:                                        
                                        dir_key_list.append(model_name+"/"+train_val+"/"+sub_class_name+"/"+sub_res_data["image_name"])        
        
        dir_list = s3.build_folder_tree(dir_key_list)
        root = dir_list        
        await s3.add_model_status_to_folders(root, user_name)
        if not dir_list:
            return api_json_response_format(False, "No folders", 404, {})
        
        return api_json_response_format(True, "Folder list", 200, root["children"])
    except Exception as error:
        write_iiot_log(1,"error in list_dir  "+str(error))
        return api_json_response_format(True, "Folder list", 200, [])


@train_bp.route('/list_dir_old', methods=['POST'])
@Authentication.token_required
async def list_dir_old():
    try:
        data = request.get_json()

        prefix = data.get('folder_path')

        dir_list = await s3.list_folder(request=request, prefix=prefix)

        if not dir_list:
            return api_json_response_format(False, "No folders", 404, {})
        
        return api_json_response_format(True, "Folder list", 200, dir_list)
    except Exception as error:
        print(error)
        return api_json_response_format(True, "Folder list", 200, [])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS


@train_bp.route('/upload_image', methods=['POST'])
@Authentication.token_required
async def upload_image():
    folder_path = request.form.get('folder_path')
    files = request.files.getlist('files')
    images_list = files

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

        try:                        
            update_folder_list = folder_path.split("/")
            if len(update_folder_list) > 2:
                sub_class_name = update_folder_list[2]
                query = "SELECT id,model_id FROM train_class_name  where model_id = (select model_id from train_model where model_name = %s ) and class_name = %s"
                value = (model_name,sub_class_name)
                res = current_app.database.execute_query(query,value)    
                if "data" in res:
                    if len(res["data"]) == 0:
                        sub_query = "INSERT INTO train_class_name (model_id, class_name) values((select model_id from train_model where model_name = %s ),%s)"            
                        sub_value = (model_name, sub_class_name)                        
                        sub_res = current_app.database.insert_query(sub_query,sub_value)
                        if "data" in sub_res:
                            image_class_id = sub_res["data"]                             
                            img_query = "INSERT INTO train_class_images (model_id, class_id,image_name) values((select model_id from train_model where model_name = %s ),%s,%s)"            
                        for image_name in images_list:                                                            
                            img_value = (model_name, image_class_id,image_name.filename)                        
                            img_res = current_app.database.insert_query(img_query,img_value)                           
                    else:
                        image_class_id = res["data"][0]["id"]
                        for image_name in images_list:                                                            
                            query = "delete FROM train_class_images WHERE model_id = (select model_id from train_model where model_name = %s ) AND class_id = %s and image_name = %s"
                            value = (model_name, str(image_class_id),image_name.filename)
                            res = current_app.database.update_query(query,value)                            
                            img_query = "INSERT INTO train_class_images (model_id, class_id,image_name) values((select model_id from train_model where model_name = %s ),%s,%s)"            
                            img_value = (model_name, image_class_id,image_name.filename)                        
                            img_res = current_app.database.insert_query(img_query,img_value)                            
                        

                        
                
        except Exception as error:
            write_iiot_log(1,"error in upload_image  "+str(error))

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
        query = "SELECT tr.status,tr.model_id FROM train_model AS tr LEFT JOIN users AS us ON tr.user_id = us.user_id WHERE us.user_name = %s AND tr.model_name = %s"        
        value = (user_name, temp_base_path)        
        res = current_app.database.execute_query(query,value)
        model_id = 0
        if res['data']:
            model_id = res['data'][0]['model_id']
            if res['data'][0]['status'] == 'trained':

                write_iiot_log(0,"[*] Model already trained")
                return api_json_response_format(True, "Model already trained", 200, {})

        # current_app.background_runner.executor.submit(
        #     current_app.background_runner.train_model_async, base_path, user_name
        # )
        write_iiot_log(0,"model id : "+str(model_id))
        current_app.background_runner.train_model_async(base_path, user_name,model_id)

        status = 'training'
        query = "UPDATE train_model  SET status = %s  WHERE model_name = %s AND user_id = (SELECT user_id FROM users WHERE user_name = %s);"
        write_iiot_log(0,query)
        write_iiot_log(0,status+" "+temp_base_path+"  "+user_name)
        value = (status, temp_base_path, user_name)
        res = current_app.database.update_query(query,value)
        if res['data'] > 0:
            print("[*] Value updated.", flush=True)
        else:
            print(f"[X] Error: str{res['message']}", flush=True)

        return api_json_response_format(True, "Model training started...", 200, {})

    except Exception as e:
        return api_json_response_format(False, f"Error: {str(e)}", 500, {})
    

@train_bp.route('/train_model_status', methods=['GET'])
def train_model_status():    
    try:
        status_flag = 1
        query = "SELECT epoch,train_acc,val_acc,message,updt,display_flag FROM model_status  order by status_id desc limit 1"
        value = ()
        # write_iiot_log(0,query)
        # data: {'epoch': 0, 'train_acc': 0, 'val_acc': 0, 'message': 'Downloading model-1/train/one/emr_screen1.png'}
        res = current_app.database.execute_query(query,value)
        # print(type(res))
        if res['data']:
            # write_iiot_log(0,str(res['data'][0]['epoch']))
            epoch = res['data'][0]['epoch']
            train_acc = res['data'][0]['train_acc']
            val_acc = res['data'][0]['val_acc']
            message = res['data'][0]['message']
            display_flag = res['data'][0]['display_flag']            
            current_updt = str(res['data'][0]['updt'])
            index = message.find("Training completed")
            if epoch == "":
                epoch = "0"                    
            
            if index >= 0:                
                message = "Training completed"                
                display_flag = 2
            if message == "Training process initiated..." or message == "images files download process started..." or message == "Downloading training and validation data..." or message == 'Training started...':                
                query = "update model_status set display_flag = -1 where message = '"+message+"' "
                value = ()
                current_app.database.update_query(query,value)
                # write_iiot_log(0,"updated status ")
                # write_iiot_log(0,res)
            res_data = {"epoch": (int(epoch)*2),"train_acc":train_acc,"val_acc":val_acc,"message":message}
            return {"success": True,"error_code": display_flag,"data": res_data }
            # return {"success": True,res_data}(True, "training process status ", display_flag, res_data)                                                        
        else:
            write_iiot_log(0,"error in train_model_status() "+str(res["message"]))   
            res_data = {"epoch": 0,"train_acc":0,"val_acc":0,"message":""}
            return {"success": True,"error_code": -1,"data": res_data }
            # return api_json_response_format(False, "error "+res["message"], 500, {})  

    except Exception as error:   
        write_iiot_log(0,"error in train_model_status() "+str(error))     
        res_data = {"epoch": 0,"train_acc":0,"val_acc":0,"message":""}
        return {"success": True,"error_code": -1,"data": res_data }
        # return api_json_response_format(False, "error in train_model_status() "+str(error), 500, {})
    
        

@train_bp.route('/train_model_progress', methods=['GET'])
def get_train_model_progress():    
    write_iiot_log(0,"first line model progress")
    def generate():        

        user_name = 'admin'
        last_message = None
        last_updt_time = ""
        write_iiot_log(0,"before get train model progress")
        while True:

            try:
                time.sleep(1)
                query = "SELECT epoch,train_acc,val_acc,message,updt FROM model_status order by updt desc limit 1"
                value = ()
                # data: {'epoch': 0, 'train_acc': 0, 'val_acc': 0, 'message': 'Downloading model-1/train/one/emr_screen1.png'}
                res = current_app.database.execute_query(query,value)
                if "data" in res:
                    pass
                    # print("data available")
                else:
                    write_iiot_log(0,"no not available")
                # print("**************")
                # print(res)
                # print(res["data"])
                # print("**************")
                model_id = 0
                if res['data']:
                    epoch = res['data'][0]['epoch']
                    train_acc = res['data'][0]['train_acc']
                    val_acc = res['data'][0]['val_acc']
                    message = res['data'][0]['message']
                    current_updt = str(res['data'][0]['updt'])
                    index = message.find("Training completed")
                    if epoch == "":
                        epoch = "0"

                    
                    if current_updt != last_updt_time:
                        last_updt_time = current_updt
                        write_iiot_log(1,f"data: {json.dumps({'epoch': (int(epoch)*2), 'train_acc': train_acc, 'val_acc': val_acc, 'message': message})}\n\n")
                        yield f"data: {json.dumps({'epoch': (int(epoch)*2), 'train_acc': train_acc, 'val_acc': val_acc, 'message': message})}\n\n"
                        # yield f"data: {json.dumps({'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'message': message})}\n\n"


                        # yield {"data": {'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'message': message}}
                        # yield f"data: {'epoch': epoch, 'train_acc': train_acc, 'val_acc': val_acc, 'message': message}"
                    
                    if index >= 0:
                        write_iiot_log(0,f"[*] Epoch 100 reached for user {user_name}, stopping stream.")
                        print(f"[*] Epoch 100 reached for user {user_name}, stopping stream.", flush=True)
                        break                                                            
                    
            except Exception as error:
                write_iiot_log(1,str(error))
                print(error)
            
            # progress = current_app.background_runner.get_progress(user_name)  
            
            # if progress and progress != last_message:
            #     last_message = progress
            #     data_dict = json.loads(progress)
            #     yield f"data: {data_dict}\n\n"
            #     print(data_dict)

            #     if  "Training completed" in data_dict['message']:
            #         print(f"[*] Epoch 100 reached for user {user_name}, stopping stream.", flush=True)
            #         current_app.background_runner.clear_progress(user_name)
            #         break
            # time.sleep(1)

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
                if temp_path[3] == "":
                    sub_class_name = temp_path[2]                    
                    query = "DELETE FROM train_class_images WHERE model_id = (SELECT model_id FROM train_model WHERE model_name = %s ) and class_id = (SELECT id FROM train_class_name WHERE class_name = %s )  "
                    value = (model_name,sub_class_name)                    
                    res = current_app.database.update_query(query,value)                    
                    query = "DELETE FROM train_class_name WHERE model_id = (SELECT model_id FROM train_model WHERE model_name = %s ) and class_name = %s "
                    value = (model_name,sub_class_name)                    
                    res = current_app.database.update_query(query,value)                                          
                else:
                    sub_class_name = temp_path[2]
                    del_image_name = temp_path[3]
                    query = "DELETE FROM train_class_images WHERE model_id = (SELECT model_id FROM train_model WHERE model_name = %s ) and class_id = (SELECT id FROM train_class_name WHERE class_name = %s ) and image_name = %s "
                    value = (model_name,sub_class_name,del_image_name)                    
                    res = current_app.database.update_query(query,value)                                    
            else: 
                query = "DELETE FROM train_class_images WHERE model_id = (SELECT model_id FROM train_model WHERE model_name = %s ) "
                value = (model_name,)
                res = current_app.database.update_query(query,value)
                query = "DELETE FROM train_class_name WHERE model_id = (SELECT model_id FROM train_model WHERE model_name = %s ) "
                value = (model_name,)
                res = current_app.database.update_query(query,value)
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
        write_iiot_log(1,"error in delete model process  "+str(e))
        print(f"[X] Error: {e}", flush=True)
        return api_json_response_format(False, f"Error: {e}", 500, {})

@train_bp.route('/get_report_data', methods=['POST'])
@Authentication.token_required
async def get_report_data():
    try:        
        data = request.get_json()        
        if "model" not in data:
            return "model name is required"
        model_name = data["model"]        
        cdn_directory = CDN_URL+"/prediction_report/"+model_name+"/"
        query = "SELECT * FROM prediction_report where model_id = (select model_id from train_model where model_name = %s) order by id"
        value = (model_name,)        
        res = current_app.database.execute_query(query,value)                
        prediction_report_data  = []
        for rs_report_data in res["data"]:            
            prediction_phase = (rs_report_data["prediction_phase"]).strip()                  
            graph_data_json = json.loads(rs_report_data["graph_data"])     
            graph_list = []
            for graph_data in graph_data_json:                
                graph_dict = {str(graph_data["Tap changer transition"]) : graph_data["Transition time AN [ms]"]}
                graph_list.append(graph_dict)
                prediction_result = json.loads(rs_report_data["prediction_result"])
            report_dict = {
                "phase" : prediction_phase,
                "phase_image" : cdn_directory+prediction_phase+"/"+rs_report_data["prediction_image"],
                "upload_image" : cdn_directory+prediction_phase+"/"+rs_report_data["upload_image"],
                "graph_data" : graph_list,
                "result" : prediction_result
                }            
            prediction_report_data.append(report_dict)                                            
        return api_json_response_format(True, "Prediction report details", 200, {"prediction_report": prediction_report_data})           
    except Exception as error:
        write_iiot_log(1,f"error in prediction report error is {str(error)}")
        return api_json_response_format(False, f"error in prediction report error is {str(error)}", 500, {})            
    
@train_bp.route('/start_predict', methods=['POST'])
@Authentication.token_required
async def start_predict():
    try:
        last_insert_id = 0
        start_time = time.time()
        folder_name = request.form.get('model')        
        phase = request.form.get('phase')        
        try:
            if phase == "R_Phase_Raise_Direction":
                query = "SELECT distinct pr.model_id,tm.model_name FROM prediction_report pr inner join train_model tm on pr.model_id = tm.model_id"
                value = ()        
                res = current_app.database.execute_query(query,value)                
                if "data" in res:
                    for rs_report_data in res["data"]:                                    
                        report_model_name = "prediction_report/"+rs_report_data["model_name"]+"/"                                                
                        response = await s3.delete_s3_object(report_model_name)
                query = "DELETE FROM prediction_report  "
                value = ()
                res = current_app.database.update_query(query,value)                        
        except Exception as error:
            write_iiot_log(0,str(error))
        # return api_json_response_format(True, "Prediction completed", 200, {"results": {}})

        prediction_images = request.files.get('prediction_image')
        upload_image = request.files.get('upload_image')
        excel_file = request.files.get('excel_file')
        if not folder_name:
            return api_json_response_format(False, "Folder not found.", 404, {})
        if not prediction_images:
            return api_json_response_format(False, "Images not found.", 404, {})
        if not excel_file:
            return api_json_response_format(False, "CSV file not found.", 404, {})
        json_file_name = "class_to_idx.json"
        temp_folder_name = folder_name.split('/')[0]
        json_file_path = f"{temp_folder_name}/{json_file_name}"
        pth_file_path = f"{temp_folder_name}/{temp_folder_name}_graph_classifier.pth"
        write_iiot_log(0,pth_file_path)
        graph_data_str = ""
        required_columns = ['Tap changer transition', 'Transition time AN [ms]']  
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
            write_iiot_log(1,"[X] Error reading Excel file: {e}")        
      
        write_iiot_log(0,"check model exist in s3")
        # Check if model files exist in S3
        is_folder_exist = await s3.get_dirs(folder_name)
        is_json_exist = await s3.get_dirs(json_file_path, file=True)
        # is_pth_exist = await s3.get_dirs(pth_file_path, file=True)
        write_iiot_log(0,"check model exist in s3 completed")

        if not is_folder_exist:
            return api_json_response_format(False, "Folder path not found.", 404, {})
        if not is_json_exist:
            return api_json_response_format(False, "Class to idx JSON file not found.", 404, {})        
        tmp_dir = 'predict'
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)        
        prediction_report = "prediction_report"
        try:
            s3_file_path = os.path.join(prediction_report,folder_name,phase)                        
            s3_file_key = os.path.join(s3_file_path,upload_image.filename)
            s3.upload_file_obj(upload_image,s3_file_key)           
            df = excel[required_columns]
            data_as_dict = df.to_dict(orient='records')            
            graph_data_str = json.dumps(data_as_dict)
            query = "delete from prediction_report where prediction_phase = %s and model_id = (SELECT model_id FROM train_model  WHERE model_name = %s)"
            value = (phase,folder_name)
            res = current_app.database.execute_query(query,value)   
            query = "INSERT INTO prediction_report (model_id, prediction_phase, prediction_image,upload_image,graph_data)  SELECT model_id, %s, %s, %s, %s FROM train_model  WHERE model_name = %s;"
            value = (phase, prediction_images.filename, upload_image.filename,graph_data_str,folder_name)
            res = current_app.database.insert_query(query,value)   
            if res["data"] > 0:
                last_insert_id = res["data"]
            else:
                write_iiot_log(1,"error in insert query error in start_predict ")
        except Exception as error:
            write_iiot_log(1,"error in start_predict process  "+str(error))        
        
        upload_path = os.path.join(tmp_dir, "uploaded_images")
        os.makedirs(upload_path, exist_ok=True)
        supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        if prediction_images.filename.lower().endswith(supported_exts):
            filename = prediction_images.filename
            try:
                prediction_images.save(os.path.join(upload_path, filename))               
                s3_file_key = os.path.join(s3_file_path,prediction_images.filename)
                s3.upload_file_to_s3(os.path.join(upload_path, filename),s3_file_key)
            except Exception as error:
                print(error)            
        write_iiot_log(0,"before Download process")
        download_s3_files(temp_folder_name, os.path.join(tmp_dir, temp_folder_name))
        write_iiot_log(0,"after Download process")

        local_json_path = os.path.join(tmp_dir, json_file_path)
        local_pth_path = os.path.join(tmp_dir, pth_file_path)
        base_path = os.path.join(tmp_dir, temp_folder_name)        
        # Load class mappings
        with open(local_json_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in sorted(idx_to_class)]

        
        write_iiot_log(0,"load model path : "+local_pth_path)
        write_iiot_log(0,"working model name : "+folder_name)
        local_pth_path = "Img_models/"+folder_name+"_graph_classifier.pth"
        write_iiot_log(0,local_pth_path)
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
                confidence = round(confidence, 2)
                results.append({
                    "filename": filename,
                    "predicted_class": pred_class,
                    "confidence": confidence,
                    "corrected": False,
                    "transition": transition,
                    "output": transition_output
                })
            except Exception as e:
                write_iiot_log(1,str(e))
                results.append({
                    "filename": filename,
                    "error": str(e)
                })

        end_time = time.time()
        elapsed = end_time - start_time        
        completed_time = round(elapsed / 60, 2)
        print("Prediction complted time is ",completed_time)
        write_iiot_log(1,"Prediction complted time is "+str(completed_time))
        result_data_str = json.dumps(results)


        try:              
            query = "update prediction_report set prediction_result = %s where id = %s"
            value = (result_data_str, str(last_insert_id))            
            res = current_app.database.update_query(query,value)               
        except Exception as error:
            write_iiot_log(1,"error in update prediction_report process  "+str(error))        

        return api_json_response_format(True, "Prediction completed", 200, {"results": results})

    except Exception as e:
        write_iiot_log(1,str(e))
        return api_json_response_format(False, f"Server error: {str(e)}", 500, {})




@train_bp.route('/correct_predictions', methods=['POST'])
@Authentication.token_required
async def correct_predictions():
    try:
        
        class_name = request.form.get('class_name')
        user_response = request.form.get('user_response')
        images = request.files.getlist('images')
        images_list = images

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
            if response.get("success"):
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

                try:                    
                    for image_name in images_list:                               
                        prediction_image_name = image_name.filename                    
                        query = "delete FROM train_class_images WHERE model_id = (select model_id from train_model where model_name = %s ) AND class_id = (select id from train_class_name where class_name = %s ) and image_name = %s"
                        value = (model_name, image_class_name,prediction_image_name)                        
                        res = current_app.database.update_query(query,value)                    
                        img_query = "INSERT INTO train_class_images (model_id, class_id,image_name) values((select model_id from train_model where model_name = %s ),(select id from train_class_name where class_name = %s ),%s)"            
                        img_value = (model_name, image_class_name,prediction_image_name)                                                
                        img_res = current_app.database.insert_query(img_query,img_value)                        
                except Exception as error:
                    write_iiot_log(1,str(error))                    

                if  res['data'] > 0:
                    print("[*] Value updated.", flush=True)
                else:
                    print(f"[X] Error: {res['message']}", flush=True)
                return api_json_response_format(True, f"Picture added to model {model_name} and train the model again", 200, {})
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


