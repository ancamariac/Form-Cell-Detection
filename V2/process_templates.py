import os
import cv2
import numpy as np
import pickle

def detect_box(image,line_min_width=15):
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,img_bin = cv2.threshold(gray_scale,60,120,cv2.THRESH_BINARY)
    kernal6h = np.ones((1,line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width,1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final=img_bin_h|img_bin_v

    _, _, stats,_ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats

def get_checkboxes(image, stats):
    result = []
    for x,y,w,h,_ in stats[1:]:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
        result.append([x,y,w,h])
    return result

def get_form_shape(checkboxes, threshold = 10):
    rows = []
    last_y = -11
    i = -1
    for box in checkboxes:
        if abs (box[1] - last_y) > threshold:
            i += 1
            last_y = box[1]
            rows.append([0])
        else:
            rows[i].append(0)
    
    return rows

def sort_checkboxes(checkboxes, form):
    pos = 0
    for row in form:
        row_len = len(row)
        srt = sorted(checkboxes[pos : pos + row_len], key=lambda x: x[0])
        checkboxes[pos : pos + row_len] = srt
        pos += row_len 
    return checkboxes

def get_red_box(image):

    lower_gray = np.array([0, 0, 205], np.uint8)
    upper_gray = np.array([60, 60, 255], np.uint8)

    mask = cv2.inRange(image, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(image, image, mask = mask)
    return img_res

class Page:
    def __init__(self, name, form, checkboxes):
        self.name = name
        self.form = form
        self.checkboxes = checkboxes

def main():

    pages = []
    for file in os.listdir('template'):
        template_path = os.path.join('template', file)
        template = cv2.imread(template_path)
        red_box = get_red_box(template)
        stats=detect_box(red_box, line_min_width=5)
        checkboxes = get_checkboxes(template, stats)
        form = get_form_shape(checkboxes) #init form
        checkboxes = sort_checkboxes(checkboxes, form)
        pages.append(Page(file, form, checkboxes))

    with open('templates.pk', 'wb') as file:
        pickle.dump(pages, file, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    main()