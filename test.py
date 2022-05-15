import os
import cv2
import numpy as np

from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

image_path='check.jpg'
image=cv2.imread(image_path)

file = Image.open("check.jpg")
str = pytesseract.image_to_string(file, lang="deu", config='-c preserve_interword_spaces=1 --psm 6')
#str = str.split('\n')
print(str)

