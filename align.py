import concurrent.futures
import functools
import multiprocessing
import skimage
from skimage import transform
from skimage.feature import match_descriptors, ORB
from skimage.filters import gaussian, laplace
from skimage.measure import ransac
import numpy as np
import os
import sys
import threadpoolctl
import tifffile
import tqdm


def align_pair(img1, img2, seed=1):
    """Worker function to align an image pair.

    Returns several structures from the descriptor matching and alignment process.

    """

    descriptor_extractor = ORB(n_keypoints=1000, fast_n=12, n_scales=1)
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors
    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    rng = np.random.default_rng(seed)

    # Enforce known constraints on the model parameters.
    def is_model_valid(model, *data):
        return np.linalg.norm(model.translation) < 80 and abs(model.rotation) < 0.005

    model, inliers = ransac(
        (keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]),
        transform.EuclideanTransform,
        min_samples=2,
        residual_threshold=10,
        max_trials=500000,
        rng=rng,
        is_model_valid=is_model_valid,
    )

    return model, keypoints1, keypoints2, matches, inliers


def warpc(img, tform):
    """Warp a standard row-major color image with an skimage transform.

    skimage's transforms use (x,y) coordinates so we need to transpose the image
    from (y,x) to (x,y), apply the transform, then untranspose the result.

    """
    return skimage.transform.warp(img.T, tform).T


def compose(tforms):
    """Compose transforms by multiplying their parameter matrices together."""
    matrix = functools.reduce(np.matmul, [t.params for t in tforms])
    tout = transform.EuclideanTransform(matrix)
    return tout


def render(i, m):
    # Transform image with its own alignment transform.
    timg = skimage.img_as_ubyte(warpc(i, m))
    return timg


if __name__ == "__main__":

    threadpoolctl.threadpool_limits(1)

    path_in = sys.argv[1]
    path_out = sys.argv[2]

    print("Loading images...")
    imgs = tifffile.imread(path_in)[:110]
    imgs_f = [gaussian(img) for img in imgs]
    print()

    models = []
    results = []

    if hasattr(os, 'sched_getaffinity'):
        n_workers = len(os.sched_getaffinity(0))
    else:
        n_workers = multiprocessing.cpu_count()
    print(f"Using {n_workers} parallel workers")
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)

    # Align successive image pairs to determine translation.
    n_pairs = len(imgs) - 1
    results = list(
        tqdm.tqdm(
            pool.map(align_pair, imgs_f[:-1], imgs_f[1:]),
            total=n_pairs,
            desc="Aligning images",
        )
    )
    for i, r in enumerate(results, 1):
        print(f"{i:2} -- Translation: {np.linalg.norm(r[0].translation)} /  Matches: {r[3].shape[0]:4} / Inliers: {r[4].sum():4}")

    # For each image, build the full transform to align it all the way back to the first one.
    models = [transform.EuclideanTransform()]
    for r in results:
        cur_model = r[0]
        models.append(compose([models[-1], cur_model]))

    imgs_out = np.empty_like(imgs)
    for i, timg in enumerate(
        tqdm.tqdm(
            pool.map(render, imgs, models),
            total=len(imgs),
            desc='Rendering frame',
        )
    ):
        imgs_out[i] = timg
    tifffile.imwrite(path_out, imgs_out, compression="adobe_deflate", predictor=True)

    pool.shutdown()

    print('\nDone')
