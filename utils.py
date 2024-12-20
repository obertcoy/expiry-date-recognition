from datetime import datetime
from constant import DATE_FORMATS

def format_to_datestring(date_str):
     
    for date_format in DATE_FORMATS:
        try:
            date_obj = datetime.strptime(date_str, date_format)
            formatted_date = date_obj.strftime("%d %B %Y")
            return formatted_date
        except ValueError:
            continue
    
    return date_str