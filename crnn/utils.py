import cv2 as cv
import numpy as np

def preprocess_image(image, resize=True):

    image_cv = image.astype(np.uint8)
    
    if len(image_cv.shape) == 2 or image_cv.shape[2] == 1:
        image_cv = cv.cvtColor(image_cv, cv.COLOR_GRAY2BGR)
    elif image_cv.shape[2] == 4:
        image_cv = cv.cvtColor(image_cv, cv.COLOR_RGBA2BGR)

    if(resize):
        image_cv = cv.resize(image_cv, (0, 0), fx=2, fy=2)  

    # denoised = cv.fastNlMeansDenoisingColored(image_cv, None, 10, 10, 7, 21)

    gray = cv.cvtColor(image_cv, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # edges = cv.Canny(blurred, 100, 200)

    thresh = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=1)
        
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # dilated = cv.dilate(thresh, kernel, iterations=1)
    # eroded = cv.erode(dilated, kernel, iterations=1)
    return closed

    edges = cv.Canny(blurred, 50, 150)

    # Perform morphological operations (optional) to clean edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))  # Change kernel size if needed
    # Use erosion to remove noise and shrink edges
    eroded = cv.erode(edges, kernel, iterations=1)
    # Optionally, use dilation to enhance edges if they are too thin
    dilated = cv.dilate(eroded, kernel, iterations=1)

    # Return the final processed image for visualization or OCR use
    return dilated


