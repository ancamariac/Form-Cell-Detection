import os
import cv2
import numpy as np
import sys
from skimage.metrics import structural_similarity
from pdf2image import convert_from_path

def detect_box(image,line_min_width=15):
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,img_bin = cv2.threshold(gray_scale,60,120,cv2.THRESH_BINARY)
    kernal6h = np.ones((1,line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width,1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final=img_bin_h|img_bin_v

    _, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats,labels

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
        ratio = percentage / checkbox_percent[pos]
        if best < ratio:
            best = ratio
            index = pos
        pos += 1
    if best <= 1:
        index += 1
        index *= -1
    return index    

def pos_to_index(form, pos):
    posx = 0
    posy = 0
    while pos > 0:
        pos -= 1
        len_aux = len(form[posx])
        if posy >= len_aux - 1:
            posy = 0
            posx += 1
        else:
            posy += 1
    return posx, posy

def get_checkboxes(image, stats):
    result = []
    for x,y,w,h,_ in stats[1:]:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
        result.append([x,y,w,h])
    return result

def get_form_shape(checkboxes, threshold = 30):
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
        index = check_ticked(image, boxes, chk_per[pos:pos+row_len])
        counter += 1
        pos += row_len
        if index >= 0:
            row[index] += 1
        else:
            print("==============")
            print(image_path)
            print("Row", counter, "needs verification")
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
    
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	return aligned

def sort_checkboxes(checkboxes, form):
    pos = 0
    for row in form:
        row_len = len(row)
        srt = sorted(checkboxes[pos : pos + row_len], key=lambda x: x[0])
        checkboxes[pos : pos + row_len] = srt
        pos += row_len 
    return checkboxes

def get_red_box(image):

    lower_gray = np.array([0, 0, 225], np.uint8)
    upper_gray = np.array([60, 60, 255], np.uint8)

    mask = cv2.inRange(image, lower_gray, upper_gray)
    img_res = cv2.bitwise_and(image, image, mask = mask)
    return img_res

def get_orig(image):
    orig_dir = 'original'
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width, _ = image.shape
    max = 0
    res = ""
    for elem in os.listdir(orig_dir):
        orig = cv2.imread(os.path.join(orig_dir, elem))
        orig = cv2.resize(orig, (width, height), interpolation = cv2.INTER_AREA)
        orig = align_images(orig, image)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

        (score, _) = structural_similarity(image_gray, orig, full=True)
        if score * 100 > max:
            max = score * 100
            res = elem
    return res

def pdf2img(pdf):
    # Store Pdf with convert_from_path function
    images = convert_from_path(pdf)
    return images

def process_page(image, form_id, page_number):
    path = f'{form_id}_{page_number}.jpg'
    print(path)

    orig_path = os.path.join('original', path)
    template_path = os.path.join('template', path)

    template = cv2.imread(template_path)
    original = cv2.imread(orig_path)

    to_scan = align_images(image, original)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    red_box = get_red_box(template)
    gray_scan = cv2.cvtColor(to_scan, cv2.COLOR_BGR2GRAY)

    stats,_ = detect_box(red_box, line_min_width=5)
    checkboxes = get_checkboxes(to_scan, stats)

    form = get_form_shape(checkboxes) #init form
    
    checkboxes = sort_checkboxes(checkboxes, form)
    checkboxes_perc = checkbox_percentage(original, checkboxes)
    
    form = update_form(gray_scan, form, checkboxes, checkboxes_perc, path) #update form
    
    return form

def main():

    args = sys.argv[1:]
    image_path = args[0]

    out_folder='outs'
    os.makedirs(out_folder, exist_ok=True)
    
    # input: pdf -> transformare in imagini
    # vedem carui formular ii corespunde dupa numarul de pagini
    # pentru fiecare pagina -> aplica alg
    
    images = pdf2img(image_path)
    forms = []
    
    i = 0
    for image in images:
        form = process_page(image, "page", i)
        i += 1
        forms.append(form)
    
    pretty_print(form)
    """
    (score, diff) = structural_similarity(original, gray_scan, full=True)
    diff = (diff * 255).astype("uint8")
    print("Image Similarity: {:.4f}%".format(score * 100))

    cv2.imshow('diff', diff)
    cv2.waitKey()
    """
    cv2.imwrite(os.path.join(out_folder,f'out_{path}'), to_scan)

if __name__ == "__main__":
    main()