from flask import Flask, request, Blueprint, current_app
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
from app.utils.response import api_json_response_format
from app.services.authentication import Authentication
from flask import jsonify




auth_bp = Blueprint("auth", __name__)

@auth_bp.route('/login',methods=['GET'])
def login():
    auth = request.authorization

    if not auth:
        return api_json_response_format(False,"Authentication is required",401,{})
    
    if not auth.username:
        return api_json_response_format(False,"Username is required",401,{})
    
    if not auth.password:
        return api_json_response_format(False,"Password is required",401,{})

    try:
        username = auth.username
        userpwd = auth.password
        password = None

        query = 'SELECT `userpwd` FROM `users` WHERE user_name = %s'
        value = (username,)
        res = current_app.database.execute_query(query,value)
        if res["success"] and res["error_code"] == 200:
            if res["data"]:
                password = res["data"][0]["userpwd"]
            else:
                print(f"Invalid Username...{username}")
                return api_json_response_format(False,str("Sorry, unable to authenticate. Invalid Username..."),401,{})
        
        if password == userpwd:
            res = current_app.authentication.get_api_user_access_token(username)
            return res
        else:
            print("Wrong Password..."+auth.password)
            return api_json_response_format(False,str("invalid credentials"),401,{})

    except Exception as e:
        print("Unable to issue api token, error : "+str(e))
        return api_json_response_format(False,"Sorry, unable to login. Error : "+str(e)+", We request you to try again.",500,{})

@auth_bp.route('/refresh',methods=['GET'])
@jwt_required(refresh=True)
def refresh():
    token_result = {}
    username = get_jwt_identity() 
    new_access_token = create_access_token(identity=username)
    query = "UPDATE `users` SET `access_token`=%s WHERE user_name = %s"
    values = (new_access_token,username,)
    res = current_app.database.update_query(query,values)
    if res['data'] > 0:
        token_result['access_token'] = new_access_token
        return api_json_response_format(True,"token issued successfully",200,token_result)
    else:
        return api_json_response_format(True,"Unable to issue token",401,{})



def get_user_details():
    pass


@auth_bp.route('/home',methods = ['GET'])
@Authentication.token_required
async def home():
    return "Welcome to DCRM"
