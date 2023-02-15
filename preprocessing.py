import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def extract_digits(img_arr, area_thresh1=75, area_thresh2=15, area_thresh3=20, show=False, image_num=126):
    processed = np.zeros(img_arr.shape)
    img_arr = np.uint8(img_arr)
    for index, img in enumerate(img_arr):
        area_thresh = area_thresh1
        if np.mean(img) < 60:
            ret,binarized = cv2.threshold(img,140,255,cv2.THRESH_BINARY)
        elif np.mean(img) >= 60 and np.mean(img) < 100:
            ret,binarized = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
            area_thresh = area_thresh2
        elif np.mean(img) > 200:
            ret,binarized = cv2.threshold(img,250,255,cv2.THRESH_BINARY)
        else:
            ret,binarized = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
            area_thresh = area_thresh3

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= area_thresh:   #keep
                result[labels == i + 1] = 255
        processed[index] = result

        if index == image_num and show:
            plt.figure(figsize=(5,5))
            plt.subplot(1,3,1)
            plt.imshow(np.uint8(255)-img, cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,2)
            plt.imshow(binarized, cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(1,3,3)
            plt.imshow(result, cmap='binary_r')
            plt.xticks([])
            plt.yticks([])
            plt.show()
            print(np.mean(img))
    
    return processed


# x_train = pd.read_pickle('data/test_max_x')
# img = x_train[123]
# img = img.astype(np.uint8)
# # 127 vs 20
# # 63 (binarize more)
# # 73 (binarize more)
# # 15 (binarize less)
# results = extract_digits(x_train, show=True, image_num=117)
# plt.imshow(results[127],  cmap='binary_r')
# plt.show()
