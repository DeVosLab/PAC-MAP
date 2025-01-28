import numpy as np
import math
import numbers
import numexpr
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import (ball, binary_opening, 
								binary_closing, remove_small_objects)
from scipy.ndimage import binary_fill_holes, median_filter
from scipy.spatial.distance import pdist
import os
import platform
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reproducibility
def set_random_seeds(seed=0):
	''' Set random seeds for reproducibility.

	Parameters
	----------
	seed : int, optional
		Random seed. The default is 0.
	
	Returns
	-------
	generator : torch.Generator
		Random number generator.

	'''
	if platform.system() == 'Windows':
		os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
	generator = torch.Generator().manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(mode=True, warn_only=True)
	return generator

def seed_worker(worker_id):
	'''Seed worker for reproducibility.

	Parameters
	----------
	worker_id : int
		Worker ID.
	'''

	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


# JSON compatibility
def make_dict_json_compatible(d):
	'''Make a dictionary JSON compatible by converting numpy arrays to lists
		and recursively converting lists and dictionaries to JSON compatible types.

	Parameters
	----------
	d : dict
		Dictionary to make JSON compatible.

	Returns
	-------
	d : dict
		JSON compatible dictionary.
	'''

	for k, v in d.items():
		if isinstance(v, np.ndarray):
			d[k] = v.tolist()
		if isinstance(v, (list, tuple)):
			d[k] = make_list_json_compatible(v)
		if isinstance(v, dict):
			d[k] = make_dict_json_compatible(v)
	return d


def make_list_json_compatible(l):
	'''Make a list JSON compatible by converting numpy arrays to lists
		and recursively converting lists and dictionaries to JSON compatible types.

	Parameters
	----------
	l : list
		List to make JSON compatible.

	Returns
	-------
	l : list
		JSON compatible list.
	'''

	for i, v in enumerate(l):
		if isinstance(v, np.ndarray):
			l[i] = v.tolist()
		if isinstance(v, (list, tuple)):
			l[i] = make_list_json_compatible(v)
		if isinstance(v, dict):
			l[i] = make_dict_json_compatible(v)
	return l


# Functions to convert between condensed and square matrix indices
def calc_row_idx(k, n):
	'''Calculate the row index of the element in the condensed matrix.

	Parameters
	----------
	k : int
		Index of the element in the condensed matrix.
	n : int
		Number of elements in the square matrix.

	Returns
	-------
	i : int
		Row index of the element in the square matrix.
	'''

	return int(math.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
	'''Calculate the number of elements in i rows of the square matrix.

	Parameters
	----------
	i : int
		Number of rows.
	n : int
		Number of elements in the square matrix.

	Returns
	-------
	int
		Number of elements in i rows of the square matrix.
	'''

	return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
	'''Calculate the column index of the element in the condensed matrix.

	Parameters
	----------
	k : int
		Index of the element in the condensed matrix.
	i : int
		Row index of the element in the square matrix.
	n : int
		Number of elements in the square matrix.

	Returns
	-------
	j : int
		Column index of the element in the square matrix.

	'''

	return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
	'''Convert the index of the element in the condensed matrix to the row and column indices in the square matrix.

	Parameters
	----------
	k : int
		Index of the element in the condensed matrix.
	n : int
		Number of elements in the square matrix.

	Returns
	-------
	i : int
		Row index of the element in the square matrix.
	j : int
		Column index of the element in the square matrix.
	'''

	i = calc_row_idx(k, n)
	j = calc_col_idx(k, i, n)
	return i, j

def square_to_condensed(i, j, n):
	'''Convert the row and column indices of the element in the square matrix to the index in the condensed matrix.

	Parameters
	----------
	i : int
		Row index of the element in the square matrix.
	j : int
		Column index of the element in the square matrix.
	n : int
		Number of elements in the square matrix.

	Returns
	-------
	k : int
		Index of the element in the condensed matrix.
	'''

	assert i != j, "no diagonal elements in condensed matrix"
	if i < j:
		i, j = j, i
	return n*j - j*(j+1)//2 + i - 1 - j


def merge_close_points(points, voxelsize=None, threshold=5):
	'''Merge points that are closer than a given threshold

	Parameters
	----------
	points : np.array
		Array of points with shape (n, ndim).
	voxelsize : list, optional
		Voxel size in um. The default is None.
	threshold : int, optional
		Distance threshold in um. The default is 5.
	
	Returns
	-------
	points : np.array
		Array of merged points.
	'''

	assert points.shape[1] == len(voxelsize), 'Number of columns in points and number of elements in voxelsize do not match'

	points = points.astype(float)
	# Merge points < threshold um apart
	points_um = points * np.array(voxelsize) if voxelsize is not None else points
	point_dists = pdist(points_um, 'euclidean')
	merge_idx = np.argwhere(point_dists < threshold)
	for idx in merge_idx:
		i, j = condensed_to_square(idx, len(points))
		points[i] = np.mean([points[i], points[j]], axis=0)
		points[j] = np.nan
	points = points[~np.isnan(points).any(axis=1)]
	return points


def get_padding(img, shape=None, multichannel=False): 
	'''Get padding to resize an image to a given shape.

	Parameters
	----------
	img : ndarray
		Image to resize.
	shape : tuple, optional
		Shape to resize the image to. The default is None.	
	multichannel : bool, optional
		Whether the image is multichannel. The default is False.	

	Returns
	-------
	padding : tuple
		Padding to resize the image to the given shape.
	'''  

	if multichannel:
		d, _, h, w = img.shape
		D, _, H, W = shape
	else:
		d, h, w = img.shape
		D, H, W = shape
	if shape is None:
		value = np.max([d, h, w])
		shape = (value,value,value)
	d_padding = (D - d) / 2
	h_padding = (H - h) / 2
	v_padding = (W - w) / 2
	low_pad = d_padding if d_padding % 1 == 0 else d_padding+0.5
	l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
	t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
	high_pad = d_padding if d_padding % 1 == 0 else d_padding-0.5
	r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
	b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
	if multichannel:
		padding = (
			(int(low_pad), int(high_pad)),
			(0, 0),
			(int(l_pad), int(r_pad)),
			(int(t_pad), int(b_pad))
			)
	else:
		padding = (
			(int(low_pad), int(high_pad)),
			(int(l_pad), int(r_pad)),
			(int(t_pad), int(b_pad))
			)
	return padding


def get_patch_box(centroid, patch_size):
	'''Get bounding box of a patch around a centroid.

	Parameters
	----------
	centroid : tuple
		Centroid of the patch.
	patch_size : int
		Size of the patch.

	Returns
	-------
	box : list
		Bounding box coordinates (zmin, rmin, cmin, zmax, rmax, cmax).
	'''

	z, row, col = centroid
	d_padding = patch_size / 2    
	h_padding = patch_size / 2
	v_padding = patch_size / 2
	zmin = (z - d_padding) if d_padding % 1 == 0 else (z - (d_padding+0.5))
	cmin = (col - h_padding) if h_padding % 1 == 0 else (col - (h_padding+0.5))
	rmin = (row - v_padding) if v_padding % 1 == 0 else (row - (v_padding+0.5))
	zmax = (z + d_padding) if d_padding % 1 == 0 else (z + (d_padding-0.5))
	cmax = (col + h_padding) if h_padding % 1 == 0 else (col + (h_padding-0.5))
	rmax = (row + v_padding) if v_padding % 1 == 0 else (row + (v_padding-0.5))
	box = [int(zmin), int(rmin), int(cmin), int(zmax), int(rmax), int(cmax)]
	return box


def crop_ROI(img, bbox, masked_patch=False, masks=None, label=None):
	'''Crop a region of interest (ROI) from an image and only keep the masked patch.

	Parameters
	----------
	img : ndarray
		Image to crop.
	bbox : list
		Bounding box coordinates (zmin, rmin, cmin, zmax, rmax, cmax).
	masked_patch : bool, optional
		Whether to mask the patch using the defined mask. The default is False.
	masks : ndarray
		Masks used to define foreground and background.
	label : int
		Label to use as foreground.
	
	Returns
	-------
	ROI : ndarray
		Cropped ROI.
	'''

	# Add channel dimension if missing
	if img.ndim ==3:
		img = np.expand_dims(img, axis=1)
	
	# Mask foreground
	if masked_patch:
		assert masks is not None, 'masks should be provided if masked_patch is True'
		assert label is not None, 'label should be provided if masked_patch is True'
		masks = np.where(masks == label, 1, 0)
		if img.shape[1] > 1:
			masks = np.expand_dims(masks, axis=1) if masks.ndim == 3 else masks
			masks = np.repeat(masks, img.shape[1], axis=1)
		img = np.where(masks, img, 0)
	
	# Crop ROI
	ROI = img[
		bbox[0]:bbox[3],
		:,
		bbox[1]:bbox[4],
		bbox[2]:bbox[5]
		]
	return ROI


def crop_stack(img, channel2use=0, padding=None, masked_patch=False, min_size=0):
	'''Crop the stack to the smallest bounding box containing the largest foreground objects.
	
	Parameters
	----------
	img : ndarray
		Stack to crop (Z, C, H, W) or (Z, H, W).
	channel2use : int, optional
		Channel to use for cropping. The default is 0.
	padding : int, optional
		Padding to add to the bounding box. The default is None.
	masked_patch : bool, optional
		Whether to crop the stack to the masked patch. The default is False.
	
	Returns
	-------
	ROI : ndarray
		Cropped stack.
	bbox : list
		Bounding box coordinates (zmin, rmin, cmin, zmax, rmax, cmax).
	foreground : ndarray
		Foreground mask.
	'''

	if padding is not None:
		assert type(padding) is int

	def getLargestCC(segmentation):
		labels = label(segmentation)
		if labels.max() == 0:
			# No objects found, return original image'
			return segmentation
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
		return largestCC

	channel_dim = 1
	if img.ndim == 3: # No channel dimension
		img = np.expand_dims(img, axis=channel_dim) # Add channel dimension (Z, C, H, W)
		channel2use=0

	# Get channel to use
	img2use = img[:, channel2use, :, :]

	# Median filter with 2D kernel
	img2use = median_filter(img2use, size=(1, 3, 3))

	# Get foreground
	foreground = img2use > threshold_otsu(img2use)

	# Morphological operations
	foreground = binary_opening(foreground, footprint=ball(1))
	foreground = binary_fill_holes(foreground)

	# Keep only objects larger than min_size of the image size
	if min_size > 0.0:
		if isinstance(min_size, float):
			min_size = int(foreground.size * min_size)
		foreground = remove_small_objects(foreground, min_size=min_size)

	# Keep only the largest object
	foreground = getLargestCC(foreground)

	# Morphological closing
	foreground = binary_closing(foreground, footprint=ball(3))

	# Median filter with 2D kernel
	foreground = median_filter(foreground, size=(1, 5, 5)).astype(bool)

	# Get bounding box
	props = regionprops_table(
		 np.where(foreground, 1, 0), # regionprops requires labeled image
		 properties=('label', 'bbox')
		 )
	bboxs = [np.round([zmin, ymin, xmin, zmax, ymax, xmax]).astype(int) for \
			zmin, ymin, xmin, zmax, ymax, xmax in \
				zip(
					props['bbox-0'],
					props['bbox-1'],
					props['bbox-2'],
					props['bbox-3'], 
					props['bbox-4'],
					props['bbox-5']
	)] # No channel dimension, is accounted for in crop_ROI
	bbox = bboxs[0]
	if padding is not None:
		bbox = [max(0, bbox[0]-padding), max(0, bbox[1]-padding), max(0, bbox[2]-padding), \
				min(img.shape[0], bbox[3]+padding), min(img.shape[2], bbox[4]+padding), min(img.shape[3], bbox[5]+padding)]
	foreground = np.expand_dims(foreground, axis=channel_dim)
	img = crop_ROI(img, bbox=bbox, masked_patch=masked_patch, masks=foreground, label=props['label'][0])
	foreground = crop_ROI(foreground, bbox=bbox, masked_patch=masked_patch, masks=foreground, label=props['label'][0], )
	return img, bbox, foreground


def rescale_voxels(img, current_voxelsize, target_voxelsize, order=None):
	'''Rescale an image to a target voxel size.

	Parameters
	----------
	img : ndarray
		Image to rescale.
	current_voxelsize : list
		Current voxel size.
	target_voxelsize : list
		Target voxel size.
	order : int, optional
		Interpolation order. The default is None.

	Returns
	-------
	img : ndarray
		Rescaled image.
	'''

	if not isinstance(img, np.ndarray):
		raise TypeError(f'Input img should be a np.ndarray, but it is a {type(img)}.')
	if not isinstance(current_voxelsize, np.ndarray):
		current_voxelsize = np.array(current_voxelsize)
	if not isinstance(target_voxelsize, np.ndarray):
		target_voxelsize = np.array(target_voxelsize)
	if current_voxelsize.size != img.ndim:
		raise ValueError((
			f'current_voxelsize should have the same number '
			'of elements as the number of dimensions of the '
			f'input image ({img.ndim}), but it has {current_voxelsize.size}'
		))
	if target_voxelsize.size != img.ndim:
		raise ValueError((
			f'target_voxelsize should have the same number '
			'of elements as the number of dimensions of the '
			f'input image ({img.ndim}), but it has {target_voxelsize.size}'
			))

	scale = current_voxelsize/target_voxelsize
	if np.all(scale == 1.0):
		return img
	else:
		return rescale(img, scale, order=order, preserve_range=True)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
	'''Normalize an array to the range [0,1] using the minimum and maximum values.

	Parameters
	----------
	x : ndarray
		Array to normalize.
	mi : float
		Minimum value.
	ma : float
		Maximum value.
	clip : bool, optional

	Returns
	-------
	x : ndarray
		Normalized array.
	'''

	if dtype is not None:
		x   = x.astype(dtype,copy=False)
		mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
		ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
		eps = dtype(eps)
	try:
		x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
	except ImportError:
		x =                   (x - mi) / ( ma - mi + eps )
	if clip:
		x = np.clip(x,0,1)
	return x


def check_len_equals_num(a, num):
	'''Check if the length of a equals num. If a is a single element, it is repeated num times.

	Parameters
	----------
	a : list
		List to check.
	num : int
		Target number of elements.

	Returns
	-------
	a : list
		List with num elements.
	'''

	if not isinstance(a, list):
		raise TypeError(f'a should be of type list, but found {type(a)}')
	if len(a) != num:
		if len(a) == 1:
			a = num*a
			return a
		else:
			raise ValueError(
				f'The number of elements in a should be ({num}) but found {len(a)}'
			)
	else:
		return a


def normalize_per_channel(img, pmins, pmaxs, channels2normalize=None):
	'''Normalize an image per channel using percentiles.

	Parameters
	----------
	img : ndarray
		Image to normalize.
	pmins : list
		Percentiles to use as minimum values.
	pmaxs : list
		Percentiles to use as maximum values.
	channels2normalize : list, optional
		Channels to normalize. The default is None.

	Returns
	-------
	img_norm : ndarray
		Normalized image.
	'''

	if not isinstance(img, np.ndarray):
		raise TypeError(f'Input img should be a numpy.ndarray, but it is a {type(img)}.')
	if not isinstance(pmins, list):
		raise TypeError(f'pmins should be of type list, but found {type(pmins)}')
	if not isinstance(pmaxs, list):
		raise TypeError(f'pmaxs should be of type list, but found {type(pmaxs)}')
	n_channels = img.shape[1]
	pmins = check_len_equals_num(pmins, n_channels)
	pmaxs = check_len_equals_num(pmaxs, n_channels)
	
	img_norm = np.empty(img.shape, dtype=float)
	for c in range(n_channels):
		if channels2normalize is not None and c not in channels2normalize:
			img_norm[:,c,] = img[:,c,]
			continue
		channel_img = img[:,c,]
		pmin_vol = np.percentile(channel_img, pmins[c])
		pmax_vol = np.percentile(channel_img, pmaxs[c])
		img_norm[:,c,] = normalize_mi_ma(
			channel_img,
			mi=pmin_vol,
			ma=pmax_vol,
			clip=True
		)
	return img_norm


def unsqueeze_to_ndim(img, n_dim):
	'''Unsqueeze an image to a given number of dimensions.

	Parameters
	----------
	img : ndarray
		Image to unsqueeze.
	n_dim : int
		Target number of dimensions.
	
	Returns
	-------
	img : ndarray
		Unsqueezed image with n_dim dimensions.	
	'''

	assert img.ndim <= n_dim, f'img.ndim should be less than or equal to n_dim, but found img.ndim={img.ndim} and n_dim={n_dim}'
	if img.ndim < n_dim:
		img = torch.unsqueeze(img,0) if isinstance(img, torch.Tensor) else np.expand_dims(img,0)
		img = unsqueeze_to_ndim(img, n_dim)
	return img


def squeeze_to_ndim(img, n_dim):
	'''Squeeze an image to a given number of dimensions.

	Parameters
	----------
	img : ndarray
		Image to squeeze.
	n_dim : int
		Target number of dimensions.

	Returns
	-------
	img : ndarray
		Squeezed image with n_dim dimensions.
	'''

	
	if img.ndim > n_dim:
		assert img.ndim >= n_dim, f'img.ndim should be greater than or equal to n_dim, but found img.ndim={img.ndim} and n_dim={n_dim}'
		assert any([s == 1 for s in img.shape]), 'img cannot be further squeezed, as it does not contain any singleton dimensions'
		img = torch.squeeze(img) if isinstance(img, torch.Tensor) else np.squeeze(img)
		img = squeeze_to_ndim(img, n_dim)
	
	return img


class AugmentContrast(object):
	'''Augment contrast of an image by multiplying it with a random factor.

	Attributes
	----------
	contrast_range : tuple
		Range of the random factor.
	channel_dim : int, optional
		Channel dimension. The default is 0.
	preserve_range : bool, optional
		Whether to preserve the original range. The default is True.
	per_channel : bool, optional
		Whether to augment each channel independently. The default is True.
	p_per_channel : float, optional
		Probability of augmenting each channel. The default is 0.5.
	
	Methods
	-------
	__call__(sample)
		Augment the contrast of the input image.
	__repr__()
		Return the class name.
	'''

	def __init__(self, contrast_range, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
		'''
		Parameters
		----------
		contrast_range : tuple
			Range of the random factor.
		channel_dim : int, optional
			Channel dimension. The default is 0.
		preserve_range : bool, optional
			Whether to preserve the original range. The default is True.
		per_channel : bool, optional
			Whether to augment each channel independently. The default is True.
		p_per_channel : float, optional
			Probability of augmenting each channel. The default is 0.5.
		'''
		self.contrast_range = contrast_range
		self.channel_dim = channel_dim
		self.preserve_range = preserve_range
		self.per_channel = per_channel
		self.p_per_channel = p_per_channel
	
	def __call__(self, sample):
		'''
		Parameters
		----------
		sample : dict
			Dictionary with 'input' and 'target' keys (containing 3D input (C, Z, Y, X) and target (Z, Y, X) images to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		
		Returns
		-------
		sample_transformed : dict
			Dictionary with 'input' and 'target' keys (each containing a 3D image (C, Z, Y, X) to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		'''
		img = sample['input']
		n_channels = img.shape[self.channel_dim]
		r1, r2 = self.contrast_range
		shape = torch.ones(img.ndim, dtype=int).tolist()
		if self.per_channel:
			shape[self.channel_dim] = n_channels
			factor = (r1 - r2) * torch.rand(shape) + r2
		else:
			factor = (r1 - r2) * torch.rand(shape) + r2
			shape[self.channel_dim] = n_channels
			factor = factor.repeat(shape)
		
		m = img.min()
		M = img.max()
		axis = list(range(img.ndim))
		axis.remove(self.channel_dim)
		augment_channel = torch.rand(shape) <= self.p_per_channel
		factor = torch.where(augment_channel, factor, torch.ones(shape))
		img = (img - img.mean(dim=axis, keepdim=True))*factor + img.mean(dim=axis, keepdim=True)
		if not self.preserve_range:
			m = img.min()
			M = img.max()
		img = img.clip(min=m, max=M)

		sample_transformed = {
			'input': img,
			'target': sample['target'],
			'input_filename': sample['input_filename'],
			'target_filename': sample['target_filename']
			}
		return sample_transformed
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"

class AugmentBrightness(object):
	'''Augment brightness of an image by adding a random factor.

	Attributes
	----------
	mu : float
		Mean of the random factor.
	sigma : float
		Standard deviation of the random factor.
	channel_dim : int, optional
		Channel dimension. The default is 0.
	preserve_range : bool, optional
		Whether to preserve the original range. The default is True.
	per_channel : bool, optional
		Whether to augment each channel independently. The default is True.
	p_per_channel : float, optional
		Probability of augmenting each channel. The default is 0.5.

	Methods
	-------
	__call__(sample)
		Augment the brightness of the input image.
	__repr__()
		Return the class name.
	'''

	def __init__(self, mu, sigma, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
		'''
		Parameters
		----------
		mu : float
			Mean of the random factor.
			sigma : float
				Standard deviation of the random factor.
			channel_dim : int, optional
				Channel dimension. The default is 0.
			preserve_range : bool, optional
				Whether to preserve the original range. The default is True.
			per_channel : bool, optional
				Whether to augment each channel independently. The default is True.
			p_per_channel : float, optional
				Probability of augmenting each channel. The default is 0.5.
		'''

		self.mu = mu
		self.sigma = sigma
		self.channel_dim = channel_dim
		self.preserve_range = preserve_range
		self.per_channel = per_channel
		self.p_per_channel = p_per_channel
	
	def __call__(self, sample):
		'''
		Parameters
		----------
		sample : dict
			Dictionary with 'input' and 'target' keys (containing 3D input (C, Z, Y, X) and target (Z, Y, X) images to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		
		Returns
		-------
		sample_transformed : dict
			Dictionary with 'input' and 'target' keys (each containing a 3D image (C, Z, Y, X) to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		'''
		img = sample['input']
		n_channels = img.shape[self.channel_dim]
		shape = torch.ones(img.ndim, dtype=int).tolist()
		if self.per_channel:
			shape[self.channel_dim] = n_channels
			rnd_nb = torch.randn(shape)*self.sigma + self.mu
		else:
			rnd_nb = torch.randn(shape)*self.sigma + self.mu
			shape[self.channel_dim] = n_channels
			rnd_nb = rnd_nb.repeat(shape)
		augment_channel = torch.rand(shape) <= self.p_per_channel
		m = img.min()
		M = img.max()
		img = img + augment_channel*rnd_nb
		if not self.preserve_range:
			m = img.min()
			M = img.max()
		img = img.clip(min=m, max=M)

		sample_transformed = {
			'input': img,
			'target': sample['target'],
			'input_filename': sample['input_filename'],
			'target_filename': sample['target_filename']
			}
		return sample_transformed
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"


class RandomFlip3D(object):
	'''Randomly flip an image along a given axis.
	
	Attributes
	----------
	axis : int
		Axis to flip the image along.
	p : float
		Probability of flipping the image.

	Methods
	-------
	__call__(sample)
		Flip the input image.
	__repr__()
		Return the class name.
	'''

	def __init__(self, axis, p=0.5):
		'''
		Parameters
		----------
		axis : int
			Axis to flip the image along.
		p : float
			Probability of flipping the image.
		'''

		if not isinstance(axis, int):
			raise TypeError('axis should be an integer')
		if not isinstance(p, (numbers.Number)) and 0 <= p <= 1:
			raise ValueError('p should be a number between 0 and 1')
		self.axis = axis
		self.p = p
		
	def __call__(self, sample):
		'''
		Parameters
		----------
		sample : dict
			Dictionary with 'input' and 'target' keys (containing 3D input (C, Z, Y, X) and target (Z, Y, X) images to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		
		Returns
		-------
		sample_transformed : dict
			Dictionary with 'input' and 'target' keys (each containing a 3D image (C, Z, Y, X) to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		'''
		sample_imgs = {
			'input': sample['input'],
			'target': sample['target']
		}
		sample_paths = {
			'input_filename': sample['input_filename'],
			'target_filename': sample['target_filename']
		}
		if torch.rand(1) < self.p:
			# Turn negative axis index in positive one
			if self.axis < 0:
				self.axis = self.axis % sample_imgs['input'].dim()
			transformed_sample = {key: value.flip(self.axis) for (key, value) in sample_imgs.items()}
		else:
			transformed_sample = {key: value for (key, value) in sample_imgs.items()}
		transformed_sample.update(sample_paths)
		return transformed_sample
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"


class RandomRescaledCrop3D:
	'''Randomly crop and resize an image to a given shape.

	Attributes
	----------
	scale_range : tuple
		Range of the random scale factor.
	shape : tuple
		Shape to resize the image to.
	anisotropic : bool
		Whether to apply anisotropic scaling.
	p : float
		Probability of cropping and resizing the image.

	Methods
	-------
	__call__(sample)
		Crop and resize the input image.
	__repr__()
		Return the class name.
	'''
	def __init__(self, scale_range=(0.5, 1.5), shape=None, anisotropic=False, p=0.5):
		assert isinstance(scale_range, (tuple, list))
		assert len(scale_range) == 2
		assert scale_range[0] < scale_range[1]
		assert shape is None or isinstance(shape, (tuple, list))
		assert shape is None or len(shape) == 3
		self.scale_range = scale_range
		self.output_shape = shape
		self.anisotropic = anisotropic
		self.p = p

	def __call__(self, sample):
		'''
		Parameters
		----------
		sample : dict
			Dictionary with 'input' and 'target' keys (containing 3D input (C, Z, Y, X) and target (Z, Y, X) images to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		
		Returns
		-------
		sample_transformed : dict
			Dictionary with 'input' and 'target' keys (each containing a 3D image (C, Z, Y, X) to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		'''
		img, target = sample['input'], sample['target']
		if torch.rand(1) <= self.p:
			output_shape = self.output_shape if self.output_shape is not None else img.shape

			# Get the input image size
			n_channels, depth, height, width = img.shape

			# Sample a random scale factor (between 0.8 and 1.2, for example)
			if self.anisotropic:
				scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1], size=3)
			else:
				scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])*np.ones(3)

			# Calculate the new size after applying the scale factor
			new_depth = int(depth * scale_factor[0])
			new_height = int(height * scale_factor[1])
			new_width = int(width * scale_factor[2])

			# Resize the image to the new size
			img = torch.nn.functional.interpolate(
				unsqueeze_to_ndim(img, 5),
				size=(new_depth, new_height, new_width),
				mode='trilinear',
				align_corners=False
				)
			
			# Resize the target to the new size
			target = torch.nn.functional.interpolate(
				unsqueeze_to_ndim(target, 5),
				size=(new_depth, new_height, new_width),
				mode='trilinear',
				align_corners=False
				)
			img = img.squeeze(0)
			target = target.squeeze(0)

			# If the scaled image is smaller than the output size, pad it
			if new_depth < output_shape[1] or new_height < output_shape[2] or new_width < output_shape[3]:
				pad_depth = max(output_shape[1] - new_depth, 0)
				pad_height = max(output_shape[2] - new_height, 0)
				pad_width = max(output_shape[3] - new_width, 0)
				img = torch.nn.functional.pad(img, (
					pad_width // 2, pad_width - pad_width // 2,
					pad_height // 2, pad_height - pad_height // 2,
					pad_depth // 2, pad_depth - pad_depth // 2
					))
				target = torch.nn.functional.pad(target, (
					pad_width // 2, pad_width - pad_width // 2,
					pad_height // 2, pad_height - pad_height // 2,
					pad_depth // 2, pad_depth - pad_depth // 2
					))

			# Randomly crop the resized image within the valid boundaries
			d_offset = random.randint(0, max(new_depth - output_shape[1], 0))
			h_offset = random.randint(0, max(new_height - output_shape[2], 0))
			w_offset = random.randint(0, max(new_width - output_shape[3], 0))

			img = img[
				:,
				d_offset:d_offset + output_shape[1],
				h_offset:h_offset + output_shape[2],
				w_offset:w_offset + output_shape[3]
				]
			target = target[
				:,
				d_offset:d_offset + output_shape[1],
				h_offset:h_offset + output_shape[2],
				w_offset:w_offset + output_shape[3]
				]

		sample_transformed = {
			'input': img,
			'target': target,
			'input_filename': sample['input_filename'],
			'target_filename': sample['target_filename']
			}

		return sample_transformed
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"


class AugmentGaussianNoise(object):
	'''Augment an image by adding Gaussian noise.

	Attributes
	----------
	mu : float
		Mean of the Gaussian noise.
	sigma : float or tuple
		Standard deviation of the Gaussian noise.
	p : float
		Probability of adding Gaussian noise.

	Methods
	-------
	__call__(sample)
		Add Gaussian noise to the input image.
	__repr__()
		Return the class name.
	'''

	def __init__(self, mu=0.0, sigma=(0.0, 0.2), p=0.5):
		'''
		Parameters
		----------
		mu : float
			Mean of the Gaussian noise.
		sigma : float or tuple
			Standard deviation of the Gaussian noise.
		p : float
			Probability of adding Gaussian noise.
		'''

		assert isinstance(mu, (numbers.Number))
		assert isinstance(sigma, (numbers.Number, tuple, list))
		assert isinstance(p, (numbers.Number)) and 0 <= p <= 1

		self.mu = mu
		self.sigma = sigma
		self.p = p
	
	def __call__(self, sample):
		'''
		Parameters
		----------
		sample : dict
			Dictionary with 'input' and 'target' keys (containing 3D input (C, Z, Y, X) and target (Z, Y, X) images to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		
		Returns
		-------
		sample_transformed : dict
			Dictionary with 'input' and 'target' keys (each containing a 3D image (C, Z, Y, X) to augment), 
			and 'input_filename' and 'target_filename' keys (containing the original filename of the input and target images).
		'''
		img = sample['input']
		if torch.rand(1) < self.p:
			if isinstance(self.sigma, (tuple, list)):
				sigma = random.uniform(self.sigma[0], self.sigma[1])
			else:
				sigma = self.sigma
			rnd_nb = torch.randn(img.shape)*sigma + self.mu
			img = img + rnd_nb
			img = (img - img.min()) / (img.max() - img.min() + 1e-8)

		sample_transformed = {
			'input': img,
			'target': sample['target'],
			'input_filename': sample['input_filename'],
			'target_filename': sample['target_filename']
			}

		return sample_transformed
	
	def __repr__(self):
		return f"{self.__class__.__name__}()"


class TverskyLoss(nn.Module):
	'''Tversky loss for imbalanced data.

	Attributes
	----------
	binarize_targets : bool
		Whether to binarize the targets.

	Methods
	-------
	forward(inputs, targets, smooth=1, beta=0.5)
		Calculate the Tversky loss.
	'''

	def __init__(self, binarize_targets=False):
		'''
		Parameters
		----------
		binarize_targets : bool
			Whether to binarize the targets.
		'''
		super(TverskyLoss, self).__init__()
		self.binarize_targets = binarize_targets

	def forward(self, inputs, targets, smooth=1, beta=0.5):
		'''
		Parameters
		----------
		inputs : torch.Tensor
			Predicted values.
		targets : torch.Tensor
			Target values.
		smooth : float, optional
			Smoothing factor. The default is 1.
		beta : float, optional
			Tversky index parameter. The default is 0.5.

		Returns
		-------
		Tversky loss
		'''
		if self.binarize_targets:
			targets = (targets > 0.0).float()    
		
		#flatten label and prediction tensors
		inputs = inputs.view(-1)
		targets = targets.view(-1)
		
		#True Positives, False Positives & False Negatives
		TP = (inputs * targets).sum()    
		FP = ((1-targets) * inputs).sum()
		FN = (targets * (1-inputs)).sum()
	   
		Tversky = (TP + smooth) / (TP + beta*FP + (1-beta)*FN + smooth)  
		
		return 1 - Tversky