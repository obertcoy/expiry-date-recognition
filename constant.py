import string

CHAR_LIST = string.ascii_letters + string.digits + ' '
CRNN_WEIGHTS = './crnn_best_model.h5'
INDEX_TO_CHAR_DICT = {idx: character for idx, character in enumerate(CHAR_LIST)}
IMAGE_SIZE = (100, 32)
INDEX_TO_CHAR_TRADITIONAL = ['', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

DATE_FORMATS = [
        "%Y %m %d",
        "%d %m %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m.%d.%Y",
        "%d.%m.%Y",
        "%Y.%m.%d",
        "%d.%m.%Y",
        "%Y/%m/%d",
        "%d.%b.%Y",
        "%b.%d.%Y",
        "%Y.%b.%d",
        "%d %b %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y %B %d",
        "%d %B %Y",
    ]
