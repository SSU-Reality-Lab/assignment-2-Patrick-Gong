import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

# TODO 8단계의 매칭이 잘된 이유를 features.py코드를 바탕으로 구체적으로 작성하시오.
# 두 매칭 방법 모두 dist = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean')로 
# 이미지 1의 각 특징점과 이미지 2의 모든 특징점 사이의 거리를 계산하고 난 다음 
# 아래의 반복문을 통해 거리가 가장 짧은 거리(L2 norm)를 찾아 매칭을 수행한다. 
# 그러나 두 방법은 최소 거리를 결과 match 배열에 저장하는 과정에서 차이가 있다.

# 우선 SSD Feature Matcher는 두 개의 특징점 사이의 거리를 비교하여 매칭을 수행한다.
# 이미지 1의 각 특징점과 이미지 2의 모든 특징점 사이의 거리를 계산한 후
# 이미지 1의 특징점 A에서 이미지 2에서 가장 가까운 특징점을 찾아 매칭한다.
#  match.distance = dist[i, int(min_dist)]으로 매칭이 완료된 거리를 저장한다.

# 반면 ratio feture matcher는 SSD feture matcher의 과정에서 후보 간의 비교를 추가한다.
# 이미지 1에서 찾은 각 특징점에 매칭될 이미지 2의 특징점 중 가장 가까운 두 특징점을 찾고
# 가장 거리가 짧은 후보 / 두 번째로 거리가 짧은 후보의 비율을 계산하고 
# 이 비율이 정해놓은 임계값보다 크면 이 거리는 저장하지 않는다. 
# SSD Feature Matcher에서는 무조건 최소 거리의 매칭을 저장하기 때문에 비슷한 거리에 있는 이미지 2의 특징점들을 무시하게 되는데
# 이 과정에서 FPR(False Positive Rate)이 증가한다는 문제가 있다. 
# 반면, Ratio Feature Matcher는 이미지 1의 각 특징점에 매칭될 이미지 2의 특징점 중 가장 가까운 두 특징점을 찾고 
# 비율 계산을 통해 두 후보를 비교하여 더 신뢰할 수 있는 매칭을 찾는다.
