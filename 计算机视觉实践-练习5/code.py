import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取左图和右图
img_left = cv2.imread('left.jpg', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('right.jpg', cv2.IMREAD_GRAYSCALE)

# 创建立体匹配对象
stereo = cv2.StereoSGBM_create(minDisparity=0,
                               numDisparities=16*5,
                               blockSize=5,
                               P1=8*3*5**2,
                               P2=32*3*5**2,
                               disp12MaxDiff=1,
                               uniquenessRatio=15,
                               speckleWindowSize=0,
                               speckleRange=2,
                               preFilterCap=63,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

# 计算视差图
disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0

# 归一化视差图以便显示
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

# 显示视差图
plt.imshow(disparity, cmap='gray')
plt.colorbar()
plt.show()
