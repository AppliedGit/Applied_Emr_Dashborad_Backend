import os
from dotenv import load_dotenv
import pymysql
from app.services.emr_logger import write_iiot_log
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
            connect_timeout=240,
            cursorclass=pymysql.cursors.DictCursor
        )
        write_iiot_log(0,"database connected..............")        

    @staticmethod
    def api_json_response_format(status, message, error_code, data):
        return {"success": status, "message": message, "error_code": error_code, "data": data}

    def execute_query(self,query, values):    
        result = []
        res = {}        
        try:
            #write_iiot_log(0,query)
            # self.connect()
            self.mysql.ping(reconnect=True)
            with self.mysql.cursor() as cursor:
                cursor.execute(query, values)                                
                result = cursor.fetchall()                
                res = self.api_json_response_format(True,"success",200,result)
            # self.connection_close()
        except pymysql.MySQLError as e:
            res = self.api_json_response_format(False,str(e),500,{})
            # print("MySQL error:", e, flush=True)
            write_iiot_log(1,str(e))
            # self.connect()  # Reconnect on error
            # return self.execute_query(query, values)
        except Exception as e:
            error = f"Error while execute query. Error : {e}"
            write_iiot_log(1,query)
            write_iiot_log(1,str(e))                        
            res = self.api_json_response_format(False,error,500,{})
        finally:                    
            return res
        
    def update_query(self,query, values):    
        res = {}        
        try:        
            #write_iiot_log(0,query) 
            # self.connect()
            self.mysql.ping(reconnect=True)
            with self.mysql.cursor() as cursor:
                cursor.execute(query, values)
                row_count = cursor.rowcount
                if row_count == 0:
                    row_count = 1
                res = self.api_json_response_format(True,"success",200,row_count)        
            # self.connection_close()
        except pymysql.IntegrityError as e:
            if e.args[0] == 1062:
                error_msg = str(e.args[1])
                value = error_msg.split("'")[1]
                error_msg = f"'{value}' already exists."
                res = self.api_json_response_format(False,error_msg,500,{})
            else:          
                error = f"Error while update query. Error : {e}"
                write_iiot_log(1,error)
                print(error, flush=True)
                res = self.api_json_response_format(False,error,500,{})
        except pymysql.MySQLError as e:
            write_iiot_log(1,query)
            write_iiot_log(1,"MySQL error:", e, flush=True)            
            self.connect()  # Reconnect on error
            return self.update_query(query, values)
        except Exception as e:
            error = f"Error while update query. Error : {e}"            
            write_iiot_log(1,error)
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
                res = self.api_json_response_format(False,error_msg,500,-1)
            else:
                error = str(e)        
                error = f"Error while inserting query. Error : {e}"                
                res = self.api_json_response_format(False,error,500,-1)
        except pymysql.MySQLError as e:            
            self.connect()  # Reconnect on error
            return self.insert_query(query, values)
        except Exception as e:
            error = f"Error while execute query. Error : {e}"            
            res = self.api_json_response_format(False,error,500,-1)
        finally:           
            return res
        
    def connection_close(self):
        try:
            self.mysql.close()
            # print("db connection closed")
        except Exception as error:
            print("Database connection close error: ",error)


