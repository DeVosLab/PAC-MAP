from warnings import warn

import numpy as np
import numbers
import scipy.ndimage as ndi
from scipy.spatial import cKDTree, distance
from skimage.feature.peak import _get_peak_mask, _exclude_border, _get_threshold


''' 
    Adapted from scikit-image/skimage/feature/peak.py and scikit-image/skimage/_shared/coord.py
    to allow for anisotropic spacing of peaks in 3D images.

    In the original code, the coordinates of peaks are found and peaks too close to each other
    are removed from the list of peaks. The distance between peaks is measured in pixel units.

    Small modifications are made to allow for anisotropic spacing of peaks in 3D images. If voxelsize
    is provided:
        - min_distance is interpreted in physical units
        - If footprint is not provided, the footprint is set based on min_distance and voxelsize. If provided
        as a single number, or a tuple, the footprint is interpreted in physical units and converted to
        voxel units. If provided as a numpy array, the footprint is interpreted as a voxel mask.
        - If exclude_border is provided and it's a bool, it is set based on min_distance and voxelsize. 
        If provided as a number or a tuple, it is interpreted in physical units and converted to
        voxel units.
        - The coordinates of peaks are converted to physical units, the distance between peaks is measured
    in physical units, and the coordinates of peaks are converted back to pixel units.
'''

def _ensure_spacing_rev(coord, spacing, p_norm, max_out, voxelsize=None):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points. If voxelsize is 
        provided, spacing is measured in physical units.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.
    voxelsize : float or sequence of floats, optional
        The voxel size along each spatial dimension. If a sequence, must be
        of length equal to the number of spatial dimensions of ``coord``.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """
    if voxelsize is not None:
        if isinstance(voxelsize, numbers.Number):
            voxelsize = np.array([voxelsize] * coord.shape[1])
        elif isinstance(voxelsize, (np.ndarray, list, tuple)):
            voxelsize = np.asarray(voxelsize)
    assert voxelsize is None or len(voxelsize) == coord.shape[1], \
        "voxelsize must be either None, a single number, or a np.array, list or tuple \
            with the same length as coord.shape[1]"

    # Convert coords to physical units
    if voxelsize is not None:
        coord = coord * voxelsize

    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)
    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    accepted_peaks_indices = set()
    rejected_peaks_indices = set()
    naccepted = 0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            # Remove the point itself
            candidates.remove(idx)

            # Remove points that have already been rejected, should be ignored
            candidates = [c for c in candidates if c not in rejected_peaks_indices]

            # Remove points that have already been accepted
            candidates = [c for c in candidates if c not in accepted_peaks_indices]

            # Reject the point if there are any other points within the spacing
            if len(candidates) > 0:
                rejected_peaks_indices.update([idx])
            else:
                accepted_peaks_indices.update([idx])
                naccepted += 1
                if max_out is not None and naccepted >= max_out:
                    break
    
    # Get the accepted peaks
    output = coord[list(accepted_peaks_indices)]
    spacing_output = spacing[list(accepted_peaks_indices)]
    if max_out is not None:
        output = output[:max_out]
        spacing_output = spacing_output[:max_out]
    
    # Convert back to voxel units
    if voxelsize is not None:
        output = output / voxelsize
    
    return output, spacing_output


def _ensure_spacing(coord, spacing, p_norm, max_out, voxelsize=None):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coord : ndarray
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points. If voxelsize is 
        provided, spacing is measured in physical units.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    max_out: int
        If not None, at most the first ``max_out`` candidates are
        returned.
    voxelsize : float or sequence of floats, optional
        The voxel size along each spatial dimension. If a sequence, must be
        of length equal to the number of spatial dimensions of ``coord``.

    Returns
    -------
    output : ndarray
        A subset of coord where a minimum spacing is guaranteed.

    """

    if voxelsize is not None:
        if isinstance(voxelsize, numbers.Number):
            voxelsize = np.array([voxelsize] * coord.shape[1])
        elif isinstance(voxelsize, (np.ndarray, list, tuple)):
            voxelsize = np.asarray(voxelsize)
    assert voxelsize is None or len(voxelsize) == coord.shape[1], \
        "voxelsize must be either None, a single number, or a np.array, list or tuple \
            with the same length as coord.shape[1]"

    # Convert coords to physical units
    if voxelsize is not None:
        coord = coord * voxelsize

    # Use KDtree to find the peaks that are too close to each other
    tree = cKDTree(coord)
    indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
    rejected_peaks_indices = set()
    naccepted = 0
    for idx, candidates in enumerate(indices):
        if idx not in rejected_peaks_indices:
            # keep current point and the points at exactly spacing from it
            candidates.remove(idx)
            dist = distance.cdist([coord[idx]],
                                  coord[candidates],
                                  distance.minkowski,
                                  p=p_norm).reshape(-1)
            candidates = [c for c, d in zip(candidates, dist)
                          if d < spacing[idx]]

            # candidates.remove(keep)
            rejected_peaks_indices.update(candidates)
            naccepted += 1
            if max_out is not None and naccepted >= max_out:
                break

    # Remove the peaks that are too close to each other
    output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)
    spacing_output = np.delete(spacing, tuple(rejected_peaks_indices), axis=0)
    if max_out is not None:
        output = output[:max_out]
        spacing_output = spacing_output[:max_out]
    
    # Convert output coords back to pixel units
    if voxelsize is not None:
        output = output / voxelsize

    return output, spacing_output


def ensure_spacing(coords, spacing=1, p_norm=np.inf, min_split_size=50,
                   max_out=None, *, max_split_size=2000, top_down=True, voxelsize=None):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """
    ensure_spacing_func = _ensure_spacing if top_down else _ensure_spacing_rev

    output = coords
    if len(coords):
        if isinstance(spacing, numbers.Number):
            spacing = spacing * np.ones(coords.shape[0])

        coords = np.atleast_2d(coords)
        if min_split_size is None:
            batch_list = [coords]
            spacing_batch_list = [spacing]
        else:
            coord_count = len(coords)
            split_idx = [min_split_size]
            split_size = min_split_size
            while coord_count - split_idx[-1] > max_split_size:
                split_size *= 2
                split_idx.append(split_idx[-1] + min(split_size,
                                                     max_split_size))
            batch_list = np.array_split(coords, split_idx)
            spacing_batch_list = np.array_split(spacing, split_idx)

        output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
        spacing_output = np.zeros(0, dtype=spacing.dtype)
        for batch, spacing_batch in zip(batch_list, spacing_batch_list):
            output, spacing_output = ensure_spacing_func(np.vstack([output, batch]),
                                     np.concatenate([spacing_output, spacing_batch], axis=0), p_norm, max_out, voxelsize=voxelsize)
            if max_out is not None and len(output) >= max_out:
                break

    return output


def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm,
                              intensity_as_spacing=False, top_down=True, cap_intensities=False,
                              voxelsize=None):
    """
    Return the highest intensity peak coordinates.
    """
    # Get coordinates of peaks
    coord = np.nonzero(mask)
    intensities = image[coord]

    # Sort peaks top-down or bottom-up
    idx_sort = np.argsort(-intensities) if top_down else np.argsort(intensities)
    coord = np.transpose(coord)[idx_sort]
    intensities = intensities[idx_sort]
    if intensity_as_spacing and cap_intensities:
        # Cap the intensities to the mean intensity
        intensities = np.minimum(intensities, intensities.mean())

    if np.isfinite(num_peaks):
        max_out = int(num_peaks)
    else:
        max_out = None

    coord = ensure_spacing(
        coord,
        spacing=min_distance if not intensity_as_spacing else intensities,
        p_norm=p_norm,
        max_out=max_out,
        top_down=top_down,
        voxelsize=voxelsize
        )

    if len(coord) > num_peaks:
        coord = coord[:num_peaks]

    return coord


def _get_excluded_border_width_vox(image, min_distance_vox, exclude_border, voxelsize=None):
    """Return border_width values in voxels relative to a min_distance if requested.

    Adapted version of skimage.feature.peak._get_excluded_border_width

    """

    if isinstance(exclude_border, bool):
        border_width_vox = tuple(min_distance_vox) if exclude_border else (0,) * image.ndim
    elif isinstance(exclude_border, int):
        if exclude_border < 0:
            raise ValueError("`exclude_border` cannot be a negative value")
        if voxelsize is not None:
            exclude_border_vox = np.ceil(exclude_border / voxelsize).astype(int)
            border_width_vox = tuple(exclude_border_vox)
        else:
            border_width_vox = (exclude_border,) * image.ndim
    elif isinstance(exclude_border, tuple):
        if len(exclude_border) != image.ndim:
            raise ValueError(
                "`exclude_border` should have the same length as the "
                "dimensionality of the image.")
        for exclude in exclude_border:
            if not isinstance(exclude, int):
                raise ValueError(
                    "`exclude_border`, when expressed as a tuple, must only "
                    "contain ints."
                )
            if exclude < 0:
                raise ValueError(
                    "`exclude_border` can not be a negative value")
        if voxelsize is not None:
            exclude_border_vox = np.ceil(np.array(exclude_border) / voxelsize).astype(int)
            border_width_vox = tuple(exclude_border_vox)
        else:
            border_width_vox = exclude_border
    else:
        raise TypeError(
            "`exclude_border` must be bool, int, or tuple with the same "
            "length as the dimensionality of the image.")

    return border_width_vox


def peak_local_max(image, min_distance=1, threshold_abs=None,
                   threshold_rel=None, exclude_border=True,
                   num_peaks=np.inf, footprint=None, labels=None,
                   num_peaks_per_label=np.inf, p_norm=np.inf, 
                   intensity_as_spacing=False, top_down=True,
                   cap_intensities=True, voxelsize=None):
    """Find peaks in an image as coordinate list.

    Peaks are the local maxima in a region of `2 * min_distance + 1`
    (i.e. peaks are separated by at least `min_distance`).

    If both `threshold_abs` and `threshold_rel` are provided, the maximum
    of the two is chosen as the minimum intensity threshold of peaks.

    .. versionchanged:: 0.18
        Prior to version 0.18, peaks of the same height within a radius of
        `min_distance` were all returned, but this could cause unexpected
        behaviour. From 0.18 onwards, an arbitrary peak within the region is
        returned. See issue gh-2592.

    Parameters
    ----------
    image : ndarray
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks. To find the
        maximum number of peaks, use `min_distance=1`. If voxelsize is
        provided, min_distance is interpreted in physical units.
    threshold_abs : float or None, optional
        Minimum intensity of peaks. By default, the absolute threshold is
        the minimum intensity of the image.
    threshold_rel : float or None, optional
        Minimum intensity of peaks, calculated as
        ``max(image) * threshold_rel``.
    exclude_border : int, tuple of ints, or bool, optional
        If positive integer, `exclude_border` excludes peaks from within
        `exclude_border`-pixels of the border of the image.
        If tuple of non-negative ints, the length of the tuple must match the
        input array's dimensionality.  Each element of the tuple will exclude
        peaks from within `exclude_border`-pixels of the border of the image
        along that dimension.
        If True, takes the `min_distance` parameter as value.
        If zero or False, peaks are identified regardless of their distance
        from the border.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` peaks based on highest peak intensity.
    footprint : ndarray of bools, optional
        If provided, `footprint == 1` represents the local region within which
        to search for peaks at every point in `image`. If provided as an integer,
        a squared footprint of side length `footprint` is used. If voxelsize is provided,
        the footprint is interpreted in physical units and scaled accordingly. 
        The side length is rounded up to the nearest integer.
    labels : ndarray of ints, optional
        If provided, each unique region `labels == value` represents a unique
        region to search for peaks. Zero is reserved for background.
    num_peaks_per_label : int, optional
        Maximum number of peaks for each label.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    intensity_as_spacing : bool, optional
        If True, the intensity of the peaks is used as the spacing between
        peaks.
    top_down : bool, optional
        If True, the peaks are sorted top-down, i.e. the highest intensity
        peaks are returned first. If False, the peaks are sorted bottom-up.
    cap_intensities : bool, optional
        If True, the intensities of the peaks are capped to the mean intensity
        of the peaks.
    voxelsize : float or tuple of floats, optional
        The size of the voxels in each dimension of the image. If a tuple is
        provided, the length must match the input array's dimensionality.
    

    Returns
    -------
    output : ndarray
        The coordinates of the peaks.

    Notes
    -----
    The peak local maximum function returns the coordinates of local peaks
    (maxima) in an image. Internally, a maximum filter is used for finding
    local maxima. This operation dilates the original image. After comparison
    of the dilated and original images, this function returns the coordinates
    of the peaks where the dilated image equals the original image.

    See also
    --------
    skimage.feature.corner_peaks

    Examples
    --------
    >>> img1 = np.zeros((7, 7))
    >>> img1[3, 4] = 1
    >>> img1[3, 2] = 1.5
    >>> img1
    array([[0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 1.5, 0. , 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. , 0. , 0. ]])

    >>> peak_local_max(img1, min_distance=1)
    array([[3, 2],
           [3, 4]])

    >>> peak_local_max(img1, min_distance=2)
    array([[3, 2]])

    >>> img2 = np.zeros((20, 20, 20))
    >>> img2[10, 10, 10] = 1
    >>> img2[15, 15, 15] = 1
    >>> peak_idx = peak_local_max(img2, exclude_border=0)
    >>> peak_idx
    array([[10, 10, 10],
           [15, 15, 15]])

    >>> peak_mask = np.zeros_like(img2, dtype=bool)
    >>> peak_mask[tuple(peak_idx.T)] = True
    >>> np.argwhere(peak_mask)
    array([[10, 10, 10],
           [15, 15, 15]])

    """
    if (footprint is None or (isinstance(footprint, np.ndarray) and footprint.size == 1)) and min_distance < 1:
        warn("When min_distance < 1, peak_local_max acts as finding "
             "image > max(threshold_abs, threshold_rel * max(image)).",
             RuntimeWarning, stacklevel=2)
    
    if voxelsize is not None:
        if isinstance(voxelsize, numbers.Number):
            voxelsize = np.array([voxelsize] * image.ndim)
        elif isinstance(voxelsize, (np.ndarray, list, tuple)):
            voxelsize = np.asarray(voxelsize)
    assert voxelsize is None or len(voxelsize) == image.ndim, \
        "voxelsize must be either None, a single number, or a np.array, list or tuple \
            with the same length as coord.shape[1]"
    
    # Convert min_distance, border_width and footprint from physical units to voxels
    if voxelsize is not None:
        min_distance_vox = np.ceil(min_distance / voxelsize).astype(int)
        if footprint is None:
            footprint_vox = None
        elif isinstance(footprint, numbers.Number):
            # Assume that the footprint is given in physical units
            footprint_vox = np.ceil(footprint / voxelsize).astype(int)
            footprint_vox = np.ones(tuple(footprint_vox), dtype=bool)
        elif isinstance(footprint, tuple):
            # Assume that the footprint is given in physical units
            assert len(footprint) == image.ndim, \
                "footprint must be either None, a single number, list or tuple \
                    with the same length as the number of image dimensions \
                    or an array with the same number of dimensions as image"
            footprint_vox = np.ceil(footprint / voxelsize).astype(int)
            footprint_vox = np.ones(tuple(footprint_vox), dtype=bool)
        elif isinstance(footprint, np.ndarray) and footprint.ndim == image.ndim:
            # Assume that the footprint is representing voxels
            footprint_vox = footprint.astype(bool)
        else:
            raise ValueError("footprint must be either None, a single number, a tuple \
                    with the same length as the number of image dimensions \
                    or an array with the same number of dimensions as image")
    else:
        min_distance_vox = (int(min_distance),) * image.ndim  # in voxels
        footprint_vox = footprint if footprint is not None else None # in voxels

    border_width_vox = _get_excluded_border_width_vox(image, min_distance_vox,
                                              exclude_border, voxelsize)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint_vox is None:
        size = 2 * min_distance_vox + 1
        footprint_vox = np.ones((size), dtype=bool)
    else:
        footprint_vox = np.asarray(footprint_vox)

    if labels is None:
        # Non maximum filter
        mask = _get_peak_mask(image, footprint_vox, threshold)

        mask = _exclude_border(mask, border_width_vox)

        # Select highest intensities (num_peaks)
        coordinates = _get_high_intensity_peaks(image, mask,
                                                num_peaks,
                                                min_distance, p_norm,
                                                intensity_as_spacing=intensity_as_spacing,
                                                top_down=top_down,
                                                cap_intensities=cap_intensities,
                                                voxelsize=voxelsize)

    else:
        _labels = _exclude_border(labels.astype(int, casting="safe"),
                                  border_width_vox)

        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks
        labels_peak_coord = []

        for label_idx, roi in enumerate(ndi.find_objects(_labels)):

            if roi is None:
                continue

            # Get roi mask
            label_mask = labels[roi] == label_idx + 1
            # Extract image roi
            img_object = image[roi].copy()
            # Ensure masked values don't affect roi's local peaks
            img_object[np.logical_not(label_mask)] = bg_val

            mask = _get_peak_mask(img_object, footprint_vox, threshold, label_mask)

            coordinates = _get_high_intensity_peaks(img_object, mask,
                                                    num_peaks_per_label,
                                                    min_distance,
                                                    p_norm,
                                                    intensity_as_spacing=intensity_as_spacing,
                                                    top_down=top_down,
                                                    cap_intensities=cap_intensities,
                                                    voxelsize=voxelsize)

            # transform coordinates in global image indices space
            for idx, s in enumerate(roi):
                coordinates[:, idx] += s.start

            labels_peak_coord.append(coordinates)

        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, 2), dtype=int)

        if len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(image, out,
                                                    num_peaks,
                                                    min_distance,
                                                    p_norm,
                                                    intensity_as_spacing=intensity_as_spacing,
                                                    top_down=top_down,
                                                    cap_intensities=cap_intensities,
                                                    voxelsize=voxelsize)

    return coordinates