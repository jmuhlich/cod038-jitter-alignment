from skimage import data
from skimage import transform
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.filters import gaussian, laplace
from skimage.measure import ransac
import matplotlib.pyplot as plt
import numpy as np
import tifffile


p = "/n/files/HiTS/lsp-data/screening-dev/Coralie/PersisterCells_melanoma/Imaging/Incucyte/20241126_Lineage_0d_7d_30minInt/Plate1/Phase1/Phase_B6_1.tif"
oimg1 = tifffile.imread(p, key=133)
oimg2 = tifffile.imread(p, key=134)

img1 = gaussian(oimg1, 1)
img2 = gaussian(oimg2, 1)

print("Compute ORB descriptors")
descriptor_extractor = ORB(n_keypoints=5000, fast_n=12, n_scales=1)
descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors
descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

print("Match descriptors")
matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

rng = np.random.default_rng(1)

# Enforce known constraints on the model parameters.
def is_model_valid(model, *data):
    return np.linalg.norm(model.translation) < 80 and model.rotation < 0.001

print("Run RANSAC")
model, inliers = ransac(
    (keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]),
    transform.EuclideanTransform,
    min_samples=2,
    residual_threshold=10,
    max_trials=500000,
    rng=rng,
    # is_model_valid=is_model_valid,
)
print(f"translation={model.translation} rotation={model.rotation}")

fig, ax = plt.subplots()

plt.gray()

plot_matches(ax, img1, img2, keypoints1, keypoints2, matches[inliers])
ax.axis('off')
ax.set_title("Original Image vs. Transformed Image")


plt.show()
