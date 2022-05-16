import os
import cv2
import numpy as np
from boxdetect import config
import matplotlib.pyplot as plt

file_name = 'check.jpg'

cfg = config.PipelinesConfig()

# important to adjust these values to match the size of boxes on your image
cfg.width_range = (10,12) # 11
cfg.height_range = (14,16) # 15

# the more scaling factors the more accurate the results but also it takes more time to processing
# too small scaling factor may cause false positives
# too big scaling factor will take a lot of processing time
cfg.scaling_factors = [2]

# w/h ratio range for boxes/rectangles filtering
cfg.wh_ratio_range = (0.5, 1.7)

# group_size_range starting from 2 will skip all the groups
# with a single box detected inside (like checkboxes)
cfg.group_size_range = (1, 1)

# num of iterations when running dilation tranformation (to engance the image)
cfg.dilation_iterations = 0

from boxdetect.pipelines import get_boxes

rects, grouping_rects, image, output_image = get_boxes(
    file_name, cfg=cfg, plot=False)

print(grouping_rects)



plt.figure(figsize=(20,20))
plt.imshow(output_image)
plt.show()