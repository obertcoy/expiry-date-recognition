import cv2 as cv
import numpy as np

def preprocess_image(image, resize=True):

    image_cv = image.astype(np.uint8)
    
    if len(image_cv.shape) == 2:
        image_cv = cv.cvtColor(image_cv, cv.COLOR_GRAY2BGR)
    elif image_cv.shape[2] == 4:
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGBA2BGR)

    if(resize):
        image_cv = cv.resize(image_cv, (0, 0), fx=2, fy=2)  

    denoised = cv.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)

    gray = cv.cvtColor(denoised, cv.COLOR_BGR2GRAY)

    _, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    return binary