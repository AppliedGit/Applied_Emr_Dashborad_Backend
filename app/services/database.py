import os
from dotenv import load_dotenv
import pymysql

load_dotenv()

class Database:
    def __init__(self):
        self.connect()

    def connect(self):
        self.mysql = pymysql.connect(
            host=os.environ.get('MYSQL_HOST'),
            user=os.environ.get('MYSQL_USER'),
            password=os.environ.get('MYSQL_PASSWORD'),
            database=os.environ.get('MYSQL_DB'),
            port=int(os.environ.get('MYSQL_PORT')),
            autocommit=True,
            connect_timeout=60,
            cursorclass=pymysql.cursors.DictCursor
        )

    @staticmethod
    def api_json_response_format(status, message, error_code, data):
        return {"success": status, "message": message, "error_code": error_code, "data": data}

    def execute_query(self,query, values):    
        result = []
        res = {}        
        try:
            self.mysql.ping(reconnect=True)
            with self.mysql.cursor() as cursor:
                cursor.execute(query, values)
                result = cursor.fetchall()
                res = self.api_json_response_format(True,"success",200,result)
        except pymysql.MySQLError as e:
            print("MySQL error:", e)
            self.connect()  # Reconnect on error
            return self.execute_query(query, values)
        except Exception as e:
            error = f"Error while execute query. Error : {e}"
            print(error)
            res = self.api_json_response_format(False,error,500,{})
        finally:                    
            return res
        
    def update_query(self,query, values):    
        res = {}        
        try:        
            self.mysql.ping(reconnect=True)
            with self.mysql.cursor() as cursor:
                cursor.execute(query, values)
                row_count = cursor.rowcount
                if row_count == 0:
                    row_count = 1
                res = self.api_json_response_format(True,"success",200,row_count)        
        except pymysql.IntegrityError as e:
            if e.args[0] == 1062:
                error_msg = str(e.args[1])
                value = error_msg.split("'")[1]
                error_msg = f"'{value}' already exists."
                res = self.api_json_response_format(False,error_msg,500,{})
            else:          
                error = f"Error while update query. Error : {e}"
                print(error)
                res = self.api_json_response_format(False,error,500,{})
        except pymysql.MySQLError as e:
            print("MySQL error:", e)
            self.connect()  # Reconnect on error
            return self.update_query(query, values)
        except Exception as e:
            error = f"Error while execute query. Error : {e}"
            print(error)
            res = self.api_json_response_format(False,error,500,{})
        finally:            
            return res
        
    def insert_query(self,query, values):    
        res = {}        
        try:      
            self.mysql.ping(reconnect=True)
            with self.mysql.cursor() as cursor:  
                cursor.execute(query, values)
                lastrowid = cursor.lastrowid        
                res = self.api_json_response_format(True,"success",200,lastrowid)
        
        except pymysql.IntegrityError as e:
            if e.args[0] == 1062:
                error_msg = str(e.args[1])
                value = error_msg.split("'")[1]
                error_msg = f"'{value}' already exists."
                res = self.api_json_response_format(False,error_msg,500,{})
            else:
                error = str(e)        
                error = f"Error while inserting query. Error : {e}"
                print(error)
                res = self.api_json_response_format(False,error,500,{})
        except pymysql.MySQLError as e:
            print("MySQL error:", e)
            self.connect()  # Reconnect on error
            return self.insert_query(query, values)
        except Exception as e:
            error = f"Error while execute query. Error : {e}"
            print(error)
            res = self.api_json_response_format(False,error,500,{})
        finally:           
            return res


