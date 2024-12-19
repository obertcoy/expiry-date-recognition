import string

CHAR_LIST = string.ascii_letters + string.digits + ' '
CRNN_WEIGHTS = './acc_76_weights.h5'
INDEX_TO_CHAR_DICT = {idx: character for idx, character in enumerate(CHAR_LIST)}
IMAGE_SIZE = (100, 32)
