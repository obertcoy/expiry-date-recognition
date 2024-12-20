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

def segmentate_horizontal(image):
    vertical_pixel_count = np.sum(image == 255, axis=0)

    images = []
    current_x = 0
    is_character = False
    for index, count in enumerate(vertical_pixel_count):
        if count == 0:
            if is_character:
                # if index - current_x > 20:
                images.append(image[:, current_x:index])
                is_character = False
            current_x = index
        else:
            is_character = True
        
    return images


def segmentate_vertical(image):
    horizontal_pixel_count = np.sum(image == 255, axis=1)

    current_y = 0
    is_character = False
    for index, count in enumerate(horizontal_pixel_count):
        if count == 0:
            if is_character:
                image = image[current_y:index, :]
                break
            current_y = index
        else:
            is_character = True

    return image

def segmented_image(image):
    images = segmentate_horizontal(image)
    
    images = [segmentate_vertical(image) for image in images]

    images = [cv.copyMakeBorder(image, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=[0, 0, 0]) for image in images]
            
    return images


DATE_IMAGE_WIDTH = 320
DATE_IMAGE_HEIGHT = 80

def traditional_preprocess(image):
    original_image = cv.resize(image, (DATE_IMAGE_WIDTH, DATE_IMAGE_HEIGHT))
    image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)

    _, th3 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    black_count = np.sum(th3 == 0)
    white_count = np.sum(th3 == 255)

    if black_count < white_count:
        image = cv.bitwise_not(th3)
    else:
        image = th3
    
    segmented_images = segmented_image(image)
    
    segmented_images = [cv.resize(image, (28, 28)) for image in segmented_images]    
    segmented_images = [image.flatten() for image in segmented_images]
    
    return segmented_images


def clean_text(text):
    return re.sub(r'[^0-9]', ' ', text).strip()