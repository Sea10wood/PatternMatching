import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image1 = cv2.imread('image/view1.jpg')
image2 = cv2.imread('image/view2.jpg')

image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image1_blur = cv2.GaussianBlur(image1_gray, (5, 5), 0)
image2_blur = cv2.GaussianBlur(image2_gray, (5, 5), 0)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1_blur, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2_blur, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

camera_matrix = np.array([
    [1000.0, 0.0, image1.shape[1] / 2],
    [0.0, 1000.0, image1.shape[0] / 2],
    [0.0, 0.0, 1.0]
])
dist_coeffs = np.array([0.1, -0.2, 0.0, 0.0, 0.0]) 

E, mask = cv2.findEssentialMat(pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

_, R, t, _ = cv2.recoverPose(E, pts1, pts2, camera_matrix)


def triangulate_points(pts1, pts2, R, t, camera_matrix):
    P1 = np.hstack((camera_matrix @ np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((camera_matrix @ R, camera_matrix @ t))
    
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d_hom[:3] / points_4d_hom[3]
    return points_3d.T

points_3d = triangulate_points(pts1, pts2, R, t, camera_matrix)

def compute_camera_positions(R, t):
    position1 = np.zeros(3)  
    position2 = -R.T @ t 
    return position1, position2.flatten()

camera_position1, camera_position2 = compute_camera_positions(R, t)

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(result_image)
    plt.title('2D Matching Results')
    plt.axis('off')

def plot_3d_points_with_cameras(points, camera_position1, camera_position2):
    fig = plt.figure()
    ax = fig.add_subplot(122, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='3D Points')
    
    ax.scatter(camera_position1[0], camera_position1[1], camera_position1[2], c='b', marker='^', s=100, label='Camera Position 1')
    ax.scatter(camera_position2[0], camera_position2[1], camera_position2[2], c='g', marker='^', s=100, label='Camera Position 2')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Points and Camera Positions')
    ax.legend()

draw_matches(image1, keypoints1, image2, keypoints2, matches)
plot_3d_points_with_cameras(points_3d, camera_position1, camera_position2)
plt.show()
