import cv2
import numpy as np

def estimate_homography(ref_image, target_image):
    # 将图像转换为灰度，以便特征检测
    gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 使用SIFT检测特征点和计算描述子
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(gray_target, None)

    # 匹配特征描述子
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_ref, descriptors_target, k=2)

    # 应用比率测试，以获得好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 如果有足够的好的匹配点，则估计单应性矩阵
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 使用RANSAC算法计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H
    else:
        return None

def align_images(ref_image, target_image):
    H = estimate_homography(ref_image, target_image)
    if H is not None:
        # 应用单应性变换
        height, width = ref_image.shape[:2]
        aligned_image = cv2.warpPerspective(target_image, H, (width, height))
        return aligned_image
    else:
        return target_image  # 如果没有足够的匹配点，返回原始图像

# 读取参考图像和目标图像
ref_image = cv2.imread('path_to_reference_image.jpg')
target_image = cv2.imread('path_to_target_image.jpg')

# 对目标图像进行单应性对齐
aligned_image = align_images(ref_image, target_image)

# 显示对齐后的图像
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 接下来可以将对齐后的图像输入到burstSR网络中
