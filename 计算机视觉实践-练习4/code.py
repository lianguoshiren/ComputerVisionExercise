import cv2
import numpy as np
import sys
from PIL import Image


def show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#读入照片
image1 = cv2.imread("./paris_images_with_warps/paris_b.jpg")
image2 = cv2.imread("./paris_images_with_warps/paris_a.jpg")
A = image1.copy()
B = image2.copy()
imageA = cv2.resize(A,(0,0),fx=1,fy=1)
imageB = cv2.resize(B,(0,0),fx=1,fy=1)

#show(" ",imageA)
#检测图片的SIFT关键特征点，计算特征描述子
def detect_describe(imag):
    sift = cv2.SIFT_create()
    (kps,features) = sift.detectAndCompute(imag,None)
    kps = np.float32([kp.pt for kp in kps])
    #返回特征点集及其对应的特征描述
    return (kps,features)

kpsA,featuresA = detect_describe(imageA)
kpsB,featuresB = detect_describe(imageB)

#建立匹配器
bf = cv2.BFMatcher()
matches = bf.knnMatch(featuresA,featuresB,2)

good = [] #存放匹配上的良好特征点对
for m in matches:
    #满足条件，保留匹配对
    if len(m)==2 and m[0].distance < m[1].distance *0.75:
        good.append((m[0].trainIdx,m[1].queryIdx))


# 筛选后的匹配对数量大于4时，计算视角变换矩阵
if len(good) > 4:
    pA = np.float32([kpsA[j] for (i,j) in good])
    pB = np.float32([kpsB[i] for (i,j) in good])
    #计算视角变换矩阵
    H,status = cv2.findHomography(pA,pB,cv2.RANSAC,4.0)

M = (matches,H,status)

if M is None:
    print("无匹配结果")
    sys.exit()

(matches, H, status) = M
result = cv2.warpPerspective(imageA,H,(imageA.shape[1]+imageB.shape[1],imageA.shape[0]))
#show('res',result)
result[0:imageB.shape[0],0:imageB.shape[1]] = imageB
# result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB[:imageB.shape[0], :imageB.shape[1]]
show('res',result)
print(result.shape)