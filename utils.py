from datetime import datetime

def format_to_datestring(str):
    
    date_obj = datetime.strptime(str, "%Y %m %d")
    formatted_date = date_obj.strftime("%d %B %Y")
    
    return formatted_date

