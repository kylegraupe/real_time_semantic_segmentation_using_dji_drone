import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from skimage.util import img_as_float
from torchcrf import CRF

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

# def apply_connected_component_filter(segmentation_mask, sigma=0.5, t=0.1, connectivity=2):
#
#     blurred_image = skimage.filters.gaussian(segmentation_mask, sigma=sigma)
#     binary_mask = blurred_image < t
#     labeled_image, count = skimage.measure.label(binary_mask,connectivity=connectivity, return_num=True)
#     return labeled_image

# def apply_crf(in_frame):
#     num_classes = 23
#     crf = CRF(num_classes)
#
#     # Process the frame through your model
#     segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL,
#                                                               settings.DEVICE).astype(np.uint8)
#
#     # Resize original frame to match segmented frame
#     in_frame_resized = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)
#
#     # Convert images to floating-point for CRF processing
#     in_frame_float = torch.tensor(img_as_float(in_frame_resized)).permute(2, 0, 1).unsqueeze(0).to(
#         settings.DEVICE)
#     segmented_frame_np_gray = torch.tensor(segmented_frame_np_gray).unsqueeze(0).to(settings.DEVICE)
#
#     # Convert segmented_frame_np_gray to float tensor and add a dimension for classes
#     segmented_frame_tensor = segmented_frame_np_gray.unsqueeze(1).float()  # Shape: [batch_size, 1, height, width]
#
#     # Convert to one-hot encoding for CRF input
#     logits = torch.nn.functional.one_hot(segmented_frame_tensor.squeeze(1).long(),
#                                          num_classes=num_classes).permute(0, 3, 1, 2).float()
#     # logits should now be in shape [batch_size, num_classes, height, width]
#
#     # Apply softmax
#     logits = torch.nn.functional.softmax(logits, dim=1)
#
#     # The CRF expects (batch_size, num_classes, height, width) shape
#     # but ensure logits is in the correct shape
#     logits = logits.squeeze(0)  # Remove batch dimension if not needed
#     logits = logits.permute(2, 1, 0).unsqueeze(0)  # Ensure shape is [batch_size, num_classes, height, width]
#
#     # Pass the segmentation logits through the CRF layer
#     crf_output = crf.decode(logits.squeeze(0))  # (height, width)
#
#     # Convert CRF output to numpy array and then to image
#     crf_output_np = np.array(crf_output).astype(np.uint8)
#     print(f'CRF Output Shape: {crf_output_np.shape}')
#
#     refined_segmentation_img_rgb = settings.COLOR_MAP[crf_output_np]
#     refined_segmentation_np_rgb = np.array(refined_segmentation_img_rgb)
#
#     print(f'Refined Segmentation Shape: {refined_segmentation_np_rgb.shape}')
#     in_frame_float = in_frame_float.squeeze(0).permute(1, 2, 0).numpy()
#     print(f'Input Frame Float Shape: {in_frame_float.shape}')
#     in_frame = in_frame_float
#     segmentation_results = refined_segmentation_np_rgb
#
#     return in_frame, segmentation_results

# def apply_active_contours(segmentation_mask, snake_points=None):
#     """ Apply active contours to refine the segmentation mask boundaries. """
#     # Ensure the mask is in RGB format
#     rgb_mask = ensure_rgb(segmentation_mask)
#
#     # Convert to grayscale
#     gray_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
#
#     # Set the initial contour points (snake initialization)
#     if snake_points is None:
#         rows, cols = gray_mask.shape
#         snake_points = np.array([[
#             [cols // 4, rows // 4],
#             [3 * cols // 4, rows // 4],
#             [3 * cols // 4, 3 * rows // 4],
#             [cols // 4, 3 * rows // 4]
#         ]]).reshape(-1, 2)  # Default to a rectangle shape
#
#     # Apply active contours (snakes)
#     refined_contour = segmentation.active_contour(gray_mask, snake_points, alpha=0.015, beta=10, gamma=0.001)
#
#     # Display the result
#     plt.figure(figsize=(7, 7))
#     plt.imshow(rgb_mask)
#     plt.plot(refined_contour[:, 1], refined_contour[:, 0], '-r', lw=3)
#     plt.show()
#
#     return refined_contour
