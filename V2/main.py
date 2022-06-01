import os
import cv2
import numpy as np
from pdf2image import convert_from_path
import pickle

def checkbox_percentage(image, checkboxes, threshold = 127):
    res = []
    for checkbox in checkboxes:
        x, y, w, h = checkbox
        low = 0
        high = 0
        for i in range(w):
            for j in range(h):
                if image[y + j][x + i] < threshold:
                    low += 1
                else:
                    high += 1
        percentage = low * 100 / (low + high)
        res.append(percentage)
    return res

def check_ticked(image, checkboxes, checkbox_percent, threshold = 127):
    best = 0
    index = -1
    pos = 0
    likelihood = 0
    counter = 0

    for checkbox in checkboxes:
        x,y,w,h = checkbox
        low = 0
        high = 0
        for i in range(w):
            for j in range(h):
                if image[y + j][x + i] < threshold:
                    low += 1
                else:
                    high += 1
        percentage = low * 100 / (low + high)
        if checkbox_percent[pos] != 0:
            ratio = percentage / checkbox_percent[pos]
        else:
            ratio = percentage

        if ratio > 1:
            counter += 1

        if best < ratio:
            best = ratio
            index = pos
        pos += 1
    if best <= 1 or counter > 1:
        index += 1
        index *= -1
    return index, counter

def pretty_print(form):
    for iterator, mini_list in enumerate(form):
        print("Question " + str(iterator + 1) + ":")
        for option, element in enumerate(mini_list):
            print("option " + chr(option + 65) + ": " + str(element))
        print("\n")

def update_form(image, form, checkboxes, chk_per, image_path):
    counter = 0
    pos = 0
    for row in form:
        row_len = len(row)
        boxes = checkboxes[pos:pos+row_len]
        index, options = check_ticked(image, boxes, chk_per[pos:pos+row_len])
        counter += 1
        pos += row_len
        if index >= 0:
            row[index] += 1
        else:
            print("==============")
            print("Page", image_path + 1)
            print("Row", counter, "needs verification", options)
            print("Best match: option", chr(index * -1 + 64))
            print("==============")
    return form

def align_images(image, template, maxFeatures=500, keepPercent=0.2):
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)

	matches = sorted(matches, key=lambda x:x.distance)
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
    
	for (i, m) in enumerate(matches):
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
    
	(H, _) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	return aligned

def getShape(name, index):
    full_name = name + str(index) + '.jpg'
    with open('templates.pk', 'rb') as file:
        pages = pickle.load(file)
    for page in pages:
        if full_name == page.name:
            return page.form, page.checkboxes
    return None, None

size_to_name = {1:"cockinskala", 2:"vas", 4:"git_german_switzerland", 13:"Rehab_Cycle"}

class Page:
    def __init__(self, name, form, checkboxes):
        self.name = name
        self.form = form
        self.checkboxes = checkboxes

def main():

    pdf_path = 'test\\VAS.pdf'
    images = convert_from_path(pdf_path)
    num_pages = len(images)

    if num_pages not in size_to_name:
        print("Invalid form")
        return

    form_name = size_to_name[num_pages]
    
    cnt = 0
    for image in images:
        print("==========================")
        print("Page", cnt + 1, '\n')
        image = np.array(image)
        orig_path = os.path.join('original', form_name + str(cnt) +'.jpg')
        original = cv2.imread(orig_path)
        image = align_images(image, original)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        form, checkboxes = getShape(form_name, cnt)
        
        checkboxes_perc = checkbox_percentage(original, checkboxes)
        form = update_form(gray_img, form, checkboxes, checkboxes_perc, cnt)
        for x,y,w,h in checkboxes:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
        cv2.imshow("Image", image)
        cv2.waitKey()
        #pretty_print(form)
        cnt += 1

if __name__ == "__main__":
    main()