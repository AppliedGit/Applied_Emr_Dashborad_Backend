import logging
from logging.handlers import RotatingFileHandler
import pytz
from datetime import datetime
# from database import Database
# from env_configuration import SOURCE_PATH

#SOURCE_PATH = "/home/applied-software/Documents/1_EMR/Applied_Emr_Dashborad_Backend-dev/log/"
SOURCE_PATH = "/app/log/"

kolkata_tz = pytz.timezone('Asia/Kolkata')
logger = logging.getLogger('emr_logger')
logger.setLevel(logging.DEBUG)
kolkata_tz = pytz.timezone('Asia/Kolkata')

# handler = RotatingFileHandler('/home/email_logger.log',maxBytes=10**6,backupCount=5)
#maxBytes=10**6   # 1 MB
handler = RotatingFileHandler(SOURCE_PATH+'emr_logger.log',maxBytes=10**6,backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
def custom_format_time(record, datefmt=None):    
    dt = datetime.fromtimestamp(record.created, kolkata_tz)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

formatter.formatTime = custom_format_time 
handler.setFormatter(formatter)
logger.addHandler(handler)



def write_iiot_log(log_code,log_msg):
    print(log_msg)
    if log_code == 0:
        logger.info(log_msg)
    else:
        logger.error(log_msg)    
    # try:
    #     db = Database()
    #     if oidn_fk > 0:
    #         query = "insert into server_logs(sid_fk,oidn_fk,msg_code,log_msg) values(%s,%s,%s,%s)"
    #         db.execute_query(query,(1,oidn_fk,0,log_msg))
    # except Exception as error:
    #     logger.error(str(error))


