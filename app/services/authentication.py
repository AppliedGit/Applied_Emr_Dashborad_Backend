from dotenv import load_dotenv
import os
from flask import request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, create_refresh_token
from .database import Database
import jwt
from functools import wraps
import datetime as dt
from flask import Flask, request, Blueprint, current_app

load_dotenv()

class Authentication:
    def __init__(self,app) -> None:
        try:
            self.JWT_SECRET_KEY = "a64f5b884efe14c61cb45126d9e6dfe1d9ee556fa4502c656e4ba2b4d2f7025c"# os.environ.get('JWT_SECRET_KEY')
            self.EXP_TIME = int(os.environ.get('EXP_TIME'))
            app.config['JWT_SECRET_KEY'] = self.JWT_SECRET_KEY
            app.config['JWT_COOKIE_SECURE'] = True
            app.config["JWT_ACCESS_TOKEN_EXPIRES"] = int(os.environ.get('EXP_TIME'))
            self.jwtmanager = JWTManager(app)
            self.database = Database()
        except Exception as e:
            print(f"[X] Exception in Authentication. Error: {e}", flush=True)
    
    def api_json_response_format(self,status,message,error_code,data):
        result_json = {"success" : status,"message" : message,"error_code" : error_code,"data": data}
        return result_json
    
    # def token_required(self):
    #     async def decorator(*args, **kwargs):
    #         if 'Authorization' in request.headers:
    #             bearer_token = request.headers['Authorization'].split(" ")
    #             if len(bearer_token) == 2:
    #                 if bearer_token[0] == "Bearer":
    #                     token = bearer_token[1]
    #                     token_auth_result = self.token_authentication(token)
    #                     if token_auth_result["status_code"] == 200:
    #                         username = token_auth_result["username"]
    #                         query = 'SELECT user_id FROM users WHERE user_name = %s'
    #                         value = (username.lower(),)
    #                         res = self.database.execute_query(query,value)
    #                         if len(res) == 0:
    #                             return (jsonify({"message": "Invalid Token"}), 401) 
    #                         else:
    #                             return f(*args, **kwargs)
                        
    #                     elif token_auth_result["status_code"] == 401:
    #                         return (jsonify({"message": "Token expired"}), 401) 
    #                     else:
    #                         return (jsonify({"message": "Invalid Token"}), 401)
    #             else:
    #                 return (jsonify({"message": "Invalid Authorization"}), 401)
    #         else:
    #             return (jsonify({"message": "Authorization required"}), 401)
    #     return decorator
    @staticmethod
    def token_required(f):
        @wraps(f)
        async def decorated(*args, **kwargs):
            if 'Authorization' not in request.headers:
                return jsonify({"message": "Authorization required"}), 401
            
            auth_header = request.headers['Authorization'].split(" ")
            if len(auth_header) != 2 or auth_header[0] != "Bearer":
                return jsonify({"message": "Invalid Authorization header"}), 401
            
            token = auth_header[1]
            auth_instance = current_app.authentication
            
            # Make token_authentication async
            token_auth_result = await auth_instance.token_authentication(token)
            
            if token_auth_result["status_code"] != 200:
                return jsonify({"message": "Invalid or expired token"}), 401
                
            username = token_auth_result["username"]
            query = 'SELECT user_id FROM users WHERE user_name = %s'
            value = (username.lower(),)
            
            # Make database query async
            res = current_app.database.execute_query(query, value)
            
            if not res or not res.get("data"):
                return jsonify({"message": "Invalid Token"}), 401
                
            return await f(*args, **kwargs)
        return decorated
    

    # def token_authentication(self,token):
    #     token_status = 0
    #     try:
    #         decoded_data = jwt.decode(jwt=token,key=self.JWT_SECRET_KEY,algorithms=['HS256'])
    #         if "sub" in decoded_data:
    #             username = decoded_data["sub"]
    #             return {"username":username,"status_code":200}
    #     except Exception as error:
    #         print(error)
    #         if str(error) == "Signature has expired" or str(error) == "Invalid header padding":
    #             token_status = 401
    #             return {"status_code":token_status}
    #         else:
    #             token_status = -1
    #     return {"status_code":token_status}
    

    async def token_authentication(self, token):
        try:
            decoded_data = jwt.decode(
                jwt=token,
                key=self.JWT_SECRET_KEY,
                algorithms=['HS256']
            )
            if "sub" in decoded_data:
                return {"username": decoded_data["sub"], "status_code": 200}
        except jwt.ExpiredSignatureError:
            return {"status_code": 401}
        except Exception as e:
            print(f"[X] Token auth error: {str(e)}", flush=True)
            return {"status_code": -1}
    
    def get_api_user_access_token(self,username):
        token_result = {}
        access_token = ''
        try:                
            access_token = create_access_token(identity=username,fresh=True)
            refresh_token = create_refresh_token(username)
            query = "UPDATE `users` SET `access_token`= %s,`refresh_token`= %s WHERE user_name = %s"
            values = (access_token,refresh_token,username,)
            res =  self.database.update_query(query,values)
            if res['data'] > 0:
                token_result['access_token'] = access_token
                token_result['refresh_token'] = refresh_token
                return self.api_json_response_format(True,"User authenticated successfully",200,token_result)
            else:
                return self.api_json_response_format(True,"Unable to authenticate",401,{})
        except Exception as error:
            print(error, flush=True)
            return self.api_json_response_format(False,"Sorry, We are unable to authenticate. Error : "+str(error)+", We request you to try again.",500,{})
        
    def get_username(self, request):

        auth_header = request.headers['Authorization'].split(" ")
        if len(auth_header) != 2 or auth_header[0] != "Bearer":
            return jsonify({"message": "Invalid Authorization header"}), 401
            
        token = auth_header[1]
        try:
            decoded_data = jwt.decode(
                jwt=token,
                key=self.JWT_SECRET_KEY,
                algorithms=['HS256']
            )
            if "sub" in decoded_data:
                return {"username": decoded_data["sub"], "status_code": 200}
        except jwt.ExpiredSignatureError:
            return {"status_code": 401}
        except Exception as e:
            print(f"[X] Token auth error: {str(e)}", flush=True)
            return {"status_code": -1}
           

        