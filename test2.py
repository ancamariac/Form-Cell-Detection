import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_box(image,line_min_width=15):
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
    kernal6h = np.ones((1,line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width,1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final=img_bin_h|img_bin_v
    final_kernel = np.ones((3,3), np.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats,labels

def imshow_components(labels):
    ### creating a hsv image, with a unique hue value for each label
    label_hue = np.uint8(179*labels/np.max(labels))
    ### making saturation and volume to be 255
    empty_channel = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, empty_channel, empty_channel])
    ### converting the hsv image to BGR image
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    ### returning the color image for visualising Connected Componenets
    return labeled_img

def check_ticked(image, checkbox, threshold = 127):
    x,y,w,h = checkbox
    low = 0
    high = 0
    for i in range(w):
        for j in range(h):
            if image[y + j][x + i] < threshold:
                low += 1
            else:
                high += 1
    if low * 100 / (low + high) > 5:
        return True
    return False

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
    height, width, _ = image.shape
    for x,y,w,h,_ in stats[2:]:
        relative_w = w * 100 / width
        relative_h = h * 100 / height
        if relative_w < 2 and relative_h < 2 and relative_w > 1 and relative_h > 1:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
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

def check_ticked(image, checkbox, threshold = 127):
    x,y,w,h = checkbox
    low = 0
    high = 0
    for i in range(w):
        for j in range(h):
            if image[y + j][x + i] < threshold:
                low += 1
            else:
                high += 1
    if low * 100 / (low + high) > 5:
        return True
    return False

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

def pretty_print(form):
    for iterator, mini_list in enumerate(form):
        print("Question " + str(iterator + 1) + ":")
        for option, element in enumerate(mini_list):
            print("option " + chr(option + 65) + ": " + str(element))
        print("\n")

def update_form(image, form, checkboxes):
    counter = 0
    for box in checkboxes:
        check = check_ticked(image, box)
        if check == True:
            posx, posy = pos_to_index(form, counter)
            form[posx][posy] += 1
        counter += 1
    return form

def align_images(image, template, maxFeatures=500, keepPercent=0.2):
	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
	method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
	matcher = cv2.DescriptorMatcher_create(method)
	matches = matcher.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

    # allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images
		# map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt
    
    # compute the homography matrix between the two sets of matched
	# points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# use the homography matrix to align the images
	(h, w) = template.shape[:2]
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned


from skimage.metrics import structural_similarity

def main():
    image_path='Capture4_comp.PNG'
    image=cv2.imread(image_path)
    stats,_=detect_box(image, line_min_width=1)

    out_folder='outs'
    os.makedirs(out_folder,exist_ok=True)

    checkboxes = get_checkboxes(image, stats)
    form = get_form_shape(checkboxes)
    image_path2 = 'Capture4.PNG'
    image2=cv2.imread(image_path2)
    aligned_im2 = align_images(image2, image)
    gray = cv2.cvtColor(aligned_im2, cv2.COLOR_BGR2GRAY)
    form = update_form(gray, form, checkboxes)
    pretty_print(form)

    orig_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(orig_gray, gray, full=True)
    diff = (diff * 255).astype("uint8")
    print("Image Similarity: {:.4f}%".format(score * 100))

    cv2.imshow('diff', diff)
    cv2.waitKey()

    cv2.imwrite(os.path.join(out_folder,f'out_{image_path}'),image)

if __name__ == "__main__":
    main()