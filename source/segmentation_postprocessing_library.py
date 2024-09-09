import cv2
import numpy as np

import skimage
import torch
from PIL import Image
from skimage.util import img_as_float
from torchcrf import CRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian



import model_inference
import settings


def ensure_rgb(segmentation_mask):
    """ Ensure the segmentation mask is in RGB format. """
    if len(segmentation_mask.shape) == 3 and segmentation_mask.shape[2] == 3:
        return segmentation_mask
    return settings.COLOR_MAP[segmentation_mask.astype(np.uint8)]

def apply_erosion(segmentation_mask):
    """
    Apply erosion to the segmentation mask to reduce the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be eroded.

    Returns:
        numpy array: The eroded segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    eroded_mask = cv2.erode(segmentation_mask_rgb, settings.EROSION_KERNEL, iterations=settings.EROSION_ITERATIONS)
    return eroded_mask.astype(np.uint8)

def apply_dilation(segmentation_mask):
    """
    Apply dilation to the segmentation mask to increase the size of the segmented objects.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be dilated.

    Returns:
        numpy array: The dilated segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    dilated_mask = cv2.dilate(segmentation_mask_rgb, settings.DILATION_KERNEL, iterations=settings.DILATION_ITERATIONS)
    return dilated_mask.astype(np.uint8)

def apply_gaussian_smoothing(segmentation_mask):
    """
    Applies Gaussian smoothing to a segmentation mask.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be smoothed.

    Returns:
        numpy array: The smoothed segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.GaussianBlur(segmentation_mask_rgb, settings.GAUSSIAN_SMOOTHING_KERNEL_SHAPE, 0)).astype(np.uint8)

def apply_median_filtering(segmentation_mask):
    """
    Applies a median filter to a segmentation mask.

    Args:
        segmentation_mask (numpy array): The segmentation mask to be filtered.

    Returns:
        numpy array: The filtered segmentation mask.
    """
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.medianBlur(segmentation_mask_rgb, settings.MEDIAN_FILTERING_KERNEL_SIZE)).astype(np.uint8)



def apply_crf(original_image, segmentation_mask):
    num_classes = settings.NUM_CLASSES

    # Initialize CRF
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], num_classes)

    # print(segmentation_mask)

    # Get unary potentials (negative log probability)
    U = unary_from_labels(segmentation_mask, num_classes, gt_prob=0.90, zero_unsure=False)
    d.setUnaryEnergy(U)

    # Add color-independent term
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Add color-dependent term
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run inference
    Q = d.inference(5)

    # Find the most probable class for each pixel
    MAP = np.argmax(Q, axis=0)

    # Convert MAP to color image using COLOR_MAP
    MAP = settings.COLOR_MAP[MAP]
    # print(f'MAP.shape: {MAP.shape}')

    # Ensure MAP is in the correct shape
    if MAP.shape[0] == original_image.shape[0] * original_image.shape[1]:
        MAP = MAP.reshape(original_image.shape[0], original_image.shape[1], 3)

    # print(f'MAP.shape: {MAP.shape}')

    return MAP