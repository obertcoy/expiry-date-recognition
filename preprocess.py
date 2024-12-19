import cv2 as cv
import numpy as np
import re
from constant import IMAGE_SIZE

def crnn_preprocess_image(image, resize=True):

    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    
    image_cv = cv.resize(image_cv, IMAGE_SIZE)

    if(resize):
        image_cv = cv.resize(image_cv, (0, 0), fx=2, fy=2)  

    gray = cv.cvtColor(image_cv, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
    
    closed = np.expand_dims(closed, axis=-1)
    
    print(closed.shape)

    return closed


def tesseract_preprocess_image(image, resize=True):
    
    image_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    
    image_cv = cv.resize(image_cv, IMAGE_SIZE)

    if(resize):
        image_cv = cv.resize(image_cv, (0, 0), fx=2, fy=2)  
        
    denoised = cv.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)

    gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary

def clean_text(text):
    return re.sub(r'[^0-9]', ' ', text).strip()