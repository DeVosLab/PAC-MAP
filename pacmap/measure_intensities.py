from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import tifffile
from skimage.draw import ellipsoid


def _ensure_zyx(mask):
    if mask.ndim == 3:
        return mask.astype(bool)
    if mask.ndim == 4 and mask.shape[1] == 1:
        return np.squeeze(mask, axis=1).astype(bool)
    raise ValueError(
        f"Mask must be ZYX or Z1YX. Got shape {mask.shape}."
    )


def _ensure_zcyx(img):
    if img.ndim == 3:
        return np.expand_dims(img, axis=1)
    if img.ndim == 4:
        return img
    raise ValueError(
        f"Image must be ZYX or ZCYX. Got shape {img.shape}."
    )


def _read_points_csv(filename):
    points_df = pd.read_csv(filename, header=0, index_col=0)
    if points_df.empty:
        return points_df, np.zeros((0, 3), dtype=int)

    coord_cols = ["axis-0", "axis-1", "axis-2"]
    if all(c in points_df.columns for c in coord_cols):
        coords = points_df[coord_cols].to_numpy()
    else:
        coords = points_df.iloc[:, :3].to_numpy()
    coords = np.rint(coords).astype(int)
    return points_df, coords


def _get_valid_foreground_points(coords, mask):
    if coords.size == 0:
        return np.array([], dtype=bool)

    shape = np.array(mask.shape)
    in_bounds = ((coords >= 0) & (coords < shape)).all(axis=1)
    in_foreground = np.zeros(coords.shape[0], dtype=bool)
    if in_bounds.any():
        c = coords[in_bounds]
        in_foreground[in_bounds] = mask[c[:, 0], c[:, 1], c[:, 2]]
    return in_bounds & in_foreground


def _build_offsets(radius_um, voxelsize):
    fp = ellipsoid(
        radius_um, radius_um, radius_um, spacing=tuple(voxelsize)
    ).astype(bool)
    center = np.array(fp.shape) // 2
    offsets = np.argwhere(fp) - center
    return offsets


def _measure_means(img, points, offsets, restrict_patch_to_mask, mask):
    z_max, c_max, y_max, x_max = img.shape
    shape = np.array([z_max, y_max, x_max])
    n_points = points.shape[0]
    out = np.full((n_points, c_max), np.nan, dtype=np.float32)

    for i, p in enumerate(points):
        vox = offsets + p[None, :]
        inside = ((vox >= 0) & (vox < shape)).all(axis=1)
        vox = vox[inside]
        if vox.shape[0] == 0:
            continue
        if restrict_patch_to_mask:
            mask_keep = mask[vox[:, 0], vox[:, 1], vox[:, 2]]
            vox = vox[mask_keep]
            if vox.shape[0] == 0:
                continue

        vals = img[vox[:, 0], :, vox[:, 1], vox[:, 2]]
        out[i, :] = vals.mean(axis=0)
    return out


def _collect_samples(depatchified_path, check_batches):
    if check_batches:
        samples = []
        for p in depatchified_path.iterdir():
            if not p.is_dir():
                continue
            if p.joinpath("input").is_dir() and p.joinpath("binary").is_dir() and p.joinpath("points").is_dir():
                samples.append(p.stem)
        return sorted(samples)
    return [""]


def _collect_common_files(points_dir, image_dir, mask_dir):
    points_files = {
        f.stem for f in points_dir.iterdir()
        if f.is_file() and f.suffix == ".csv" and not f.stem.startswith(".")
    }
    image_files = {
        f.stem for f in image_dir.iterdir()
        if f.is_file() and f.suffix == ".tif" and not f.stem.startswith(".")
    }
    mask_files = {
        f.stem for f in mask_dir.iterdir()
        if f.is_file() and f.suffix == ".tif" and not f.stem.startswith(".")
    }
    return sorted(list(points_files & image_files & mask_files))


def main(**kwargs):
    depatchified_path = Path(kwargs["depatchified_path"])
    output_path = Path(kwargs["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    voxelsize = np.asarray(kwargs["voxelsize"], dtype=float)
    if voxelsize.shape != (3,):
        raise ValueError("voxelsize must be defined as 3 values: Z Y X")

    offsets = _build_offsets(kwargs["radius_um"], voxelsize)
    samples = _collect_samples(depatchified_path, kwargs["check_batches"])

    for sample in tqdm(samples, desc="Measuring patch intensities", colour="green"):
        sample_root = depatchified_path / sample if sample else depatchified_path
        points_dir = sample_root / "points"
        image_dir = sample_root / "input"
        mask_dir = sample_root / "binary"
        if not (points_dir.is_dir() and image_dir.is_dir() and mask_dir.is_dir()):
            continue
        files = _collect_common_files(points_dir, image_dir, mask_dir)

        out_dir = output_path / sample if sample else output_path
        out_dir.mkdir(parents=True, exist_ok=True)

        for stem in tqdm(files, desc=f"Sample {sample or '.'}", colour="blue", leave=False):
            points_df, coords = _read_points_csv(points_dir / f"{stem}.csv")
            if points_df.empty:
                points_df.to_csv(out_dir / f"{stem}.csv", index=True, header=True)
                continue

            img = _ensure_zcyx(tifffile.imread(image_dir / f"{stem}.tif").astype(np.float32))
            mask = _ensure_zyx(tifffile.imread(mask_dir / f"{stem}.tif"))
            valid = _get_valid_foreground_points(coords, mask)
            points_df = points_df.loc[valid].copy().reset_index(drop=True)
            coords = coords[valid]

            if coords.shape[0] == 0:
                points_df.to_csv(out_dir / f"{stem}.csv", index=True, header=True)
                continue

            means = _measure_means(
                img=img,
                points=coords,
                offsets=offsets,
                restrict_patch_to_mask=False,
                mask=mask,
            )

            for c in range(means.shape[1]):
                col = f"mean_intensity_ch{c}"
                points_df[col] = means[:, c]

            points_df.to_csv(out_dir / f"{stem}.csv", index=True, header=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--depatchified_path", type=str, required=True,
                        help="Path to depatchify output root containing sample/input,binary,points folders")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output folder for augmented points CSV files")
    parser.add_argument("--voxelsize", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                        help="Voxel size in physical units as Z Y X")
    parser.add_argument("--radius_um", type=float, required=True,
                        help="Physical radius of spherical patch in same unit as voxelsize")
    parser.add_argument("--check_batches", type=bool, default=True,
                        help="If True, depatchified_path contains sample subfolders")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
