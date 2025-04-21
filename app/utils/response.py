def api_json_response_format(status, message, error_code, data):
    return {
        "success": status,
        "message": message,
        "error_code": error_code,
        "data": data
    }