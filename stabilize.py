import argparse
import concurrent.futures
import functools
import itertools
import io
import multiprocessing
import skimage
from skimage import transform
from skimage.feature import match_descriptors, ORB
from skimage.filters import gaussian, laplace
from skimage.measure import ransac
import numpy as np
import os
import pathlib
import re
import sys
import threadpoolctl
import tifffile
import tqdm
import warnings
from xml.etree import ElementTree


def align_pair(img1, img2, seed=1):
    """Worker function to align an image pair.

    Returns several structures from the descriptor matching and alignment process.

    """

    descriptor_extractor = ORB(n_keypoints=1000, fast_n=12, n_scales=1)
    try:
        descriptor_extractor.detect_and_extract(img1)
        keypoints1 = descriptor_extractor.keypoints
        descriptors1 = descriptor_extractor.descriptors
        descriptor_extractor.detect_and_extract(img2)
        keypoints2 = descriptor_extractor.keypoints
        descriptors2 = descriptor_extractor.descriptors
    except RuntimeError:
        return None, None, None, None, None
    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    rng = np.random.default_rng(seed)

    # Enforce known constraints on the model parameters.
    def is_model_valid(model, *data):
        return np.linalg.norm(model.translation) < 80 and abs(model.rotation) < 0.005

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="skimage.measure.fit", message=r"^Estimated model is not valid")
        warnings.filterwarnings("ignore", module="skimage.measure.fit", message=r"^No inliers found")
        model, inliers = ransac(
            (keypoints1[matches[:, 0]], keypoints2[matches[:, 1]]),
            transform.EuclideanTransform,
            min_samples=2,
            residual_threshold=10,
            max_trials=50000,
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


def render(img, tform):
    # Transform multi-channel image with its own alignment transform.
    timg = np.array([skimage.img_as_ubyte(warpc(ch, tform)) for ch in img])
    return timg


def align_movie(phase_path, other_paths, output_path, pool, verbose):

    imgs = tifffile.imread(phase_path)
    imgs_f = [gaussian(img) for img in imgs]

    models = []
    results = []

    # Align successive image pairs to determine translation.
    n_pairs = len(imgs) - 1
    results = list(
        tqdm.tqdm(
            pool.map(align_pair, imgs_f[:-1], imgs_f[1:]),
            total=n_pairs,
            desc="Aligning images",
        )
    )
    del imgs_f
    if verbose:
        for i, (mo, k1, k2, ma, inl) in enumerate(results, 1):
            if mo:
                print(f"{i:2} -- Translation: {np.linalg.norm(mo.translation)} / Rotation: {mo.rotation:g} / Matches: {ma.shape[0]:4} / Inliers: {inl.sum():4}")
            else:
                print(f"{i:2} -- Failed to converge")

    # For each image, build the full transform to align it all the way back to the first one.
    models = [transform.EuclideanTransform()]
    for r in results:
        # If model fitting failed for this frame, just keep the previous frame's model.
        if r[0]:
            cur_model = r[0]
        models.append(compose([models[-1], cur_model]))

    n_channels = len(other_paths) + 1
    imgs = np.pad(
        imgs[:, None, :, :],
        ((0, 0), (0, n_channels - 1), (0, 0), (0, 0)),
    )
    for i, p in enumerate(other_paths.values()):
        imgs[:, i + 1] = np.clip(tifffile.imread(p), 0, 255)
    imgs_out = np.empty_like(imgs)
    for i, timg in enumerate(
        tqdm.tqdm(
            pool.map(render, imgs, models),
            total=len(imgs),
            desc='Rendering frame',
        )
    ):
        imgs_out[i] = timg

    tiff = tifffile.TiffFile(phase_path)
    tree = ElementTree.parse(io.StringIO(tiff.pages[0].description))
    pixel_size = [
        float(tree.find(f"./PlaneInfo/prop[@id='spatial-calibration-{dim}']").attrib["value"])
        for dim in ("x", "y")
    ]
    metadata = {
        "axes": "TCYX",
        "PhysicalSizeX": pixel_size[0],
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size[1],
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": ["Phase"] + list(other_paths)},
    }
    print(f"Writing output movie: {output_path.resolve()}")
    tifffile.imwrite(
        output_path, imgs_out, metadata=metadata, compression="adobe_deflate", predictor=True
    )


def extract_path_metadata(p):
    groups = re.match(r"([^_]+)_([^_]+)_([^_]+)$", p.stem).groups()
    return groups[1:] + (groups[0],)


if __name__ == "__main__":

    threadpoolctl.threadpool_limits(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=pathlib.Path, required=True)
    parser.add_argument("-o", "--output-path", type=pathlib.Path, required=True)
    parser.add_argument("-n", "--num-workers", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if not args.input_path.exists():
        print(f"ERROR: input_path does not exist: {args.input_path.resolve()}")
        sys.exit(1)
    if not args.output_path.exists():
        print(f"ERROR: output_path does not exist: {args.output_path.resolve()}")
        sys.exit(1)
    if list(args.output_path.iterdir()):
        print(f"WARNING: output_path contains files, contents may be overwritten: {args.output_path.resolve()}")

    if args.num_workers:
        n_workers = args.num_workers
    elif hasattr(os, 'sched_getaffinity'):
        n_workers = len(os.sched_getaffinity(0))
    else:
        n_workers = multiprocessing.cpu_count()
    print(f"Using {n_workers} parallel workers")
    print()
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)

    in_paths = {extract_path_metadata(p): p for p in args.input_path.glob("*_*_*.tif")}
    for group, meta in itertools.groupby(sorted(in_paths), key=lambda x: x[:2]):
        stem = f"{group[0]}_{group[1]}"
        out_name = f"{stem}.ome.tif"
        meta = list(meta)
        phase_idx = [m[2] == "Phase" for m in meta].index(True)
        phase_path = in_paths[meta.pop(phase_idx)]
        other_paths = {m[2]: in_paths[m] for m in sorted(meta, key=lambda x: x[2])}
        print(f"Processing {stem}\n==========")
        align_movie(phase_path, other_paths, args.output_path / out_name, pool, args.verbose)
        print()

    pool.shutdown()
    print('\nDone')
