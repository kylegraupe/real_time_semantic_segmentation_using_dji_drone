import torchcrf
from torchcrf import CRF
from skimage.util import img_as_float
import model_inference
import cv2
import time
import ffmpeg
import numpy as np
import cv2
from PIL import Image
import settings
import model_inference
import torch
import torchcrf
import cv2
import numpy as np
import pandas as pd
import numpy as np
import cv2


def ensure_rgb(segmentation_mask):
    """ Ensure the segmentation mask is in RGB format. """
    if len(segmentation_mask.shape) == 3 and segmentation_mask.shape[2] == 3:
        return segmentation_mask
    return settings.COLOR_MAP[segmentation_mask.astype(np.uint8)]

def apply_erosion(segmentation_mask):
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    eroded_mask = cv2.erode(segmentation_mask_rgb, kernel, iterations=1)
    return eroded_mask.astype(np.uint8)

def apply_dilation(segmentation_mask):
    kernel = np.ones((5, 5), np.uint8)  # Define a kernel
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    dilated_mask = cv2.dilate(segmentation_mask_rgb, kernel, iterations=1)
    return dilated_mask.astype(np.uint8)

def apply_gaussian_smoothing(segmentation_mask):
    segmentation_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.GaussianBlur(segmentation_mask_rgb, (5, 5), 0)).astype(np.uint8)

def apply_median_filtering(segmentation_mask):
    segmantion_mask_rgb = ensure_rgb(segmentation_mask)
    return np.array(cv2.medianBlur(segmantion_mask_rgb, 5)).astype(np.uint8)




def apply_crf(in_frame):
    num_classes = 23
    crf = CRF(num_classes)

    # Process the frame through your model
    segmented_frame_np_gray = model_inference.image_to_tensor(Image.fromarray(in_frame), settings.MODEL,
                                                              settings.DEVICE).astype(np.uint8)
    # segmented_frame_img_rgb = settings.COLOR_MAP[segmented_frame_np_gray]
    # segmented_frame_np_rgb = np.array(segmented_frame_img_rgb)

    # Resize original frame to match segmented frame
    in_frame_resized = cv2.resize(in_frame, (1280, 704), interpolation=cv2.INTER_NEAREST)

    # Convert images to floating-point for CRF processing
    in_frame_float = torch.tensor(img_as_float(in_frame_resized)).permute(2, 0, 1).unsqueeze(0).to(
        settings.DEVICE)
    segmented_frame_np_gray = torch.tensor(segmented_frame_np_gray).unsqueeze(0).to(settings.DEVICE)

    # Convert segmented_frame_np_gray to float tensor and add a dimension for classes
    segmented_frame_tensor = segmented_frame_np_gray.unsqueeze(1).float()  # Shape: [batch_size, 1, height, width]

    # Convert to one-hot encoding for CRF input
    logits = torch.nn.functional.one_hot(segmented_frame_tensor.squeeze(1).long(),
                                         num_classes=num_classes).permute(0, 3, 1, 2).float()
    # logits should now be in shape [batch_size, num_classes, height, width]

    # Apply softmax
    logits = torch.nn.functional.softmax(logits, dim=1)

    # The CRF expects (batch_size, num_classes, height, width) shape
    # but ensure logits is in the correct shape
    logits = logits.squeeze(0)  # Remove batch dimension if not needed
    logits = logits.permute(2, 1, 0).unsqueeze(0)  # Ensure shape is [batch_size, num_classes, height, width]

    # Pass the segmentation logits through the CRF layer
    crf_output = crf.decode(logits.squeeze(0))  # (height, width)

    # Convert CRF output to numpy array and then to image
    crf_output_np = np.array(crf_output).astype(np.uint8)
    print(f'CRF Output Shape: {crf_output_np.shape}')

    refined_segmentation_img_rgb = settings.COLOR_MAP[crf_output_np]
    refined_segmentation_np_rgb = np.array(refined_segmentation_img_rgb)

    print(f'Refined Segmentation Shape: {refined_segmentation_np_rgb.shape}')
    in_frame_float = in_frame_float.squeeze(0).permute(1, 2, 0).numpy()
    print(f'Input Frame Float Shape: {in_frame_float.shape}')
    in_frame = in_frame_float
    segmentation_results = refined_segmentation_np_rgb

    return in_frame, segmentation_results