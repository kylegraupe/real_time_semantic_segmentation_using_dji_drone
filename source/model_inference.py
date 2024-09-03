# """
# This script contains the code for the model inference.
#
# Video frames are passed to the model and the predicted RGB mask is returned.
# """
#
# # public libraries
# from torchvision import transforms
# import torch
# import time
#
#
# def load_segmentation_model(model_path):
#     """
#     Load a trained segmentation model from a file.
#
#     Args:
#         model_path (str): Path to the model file (.pt).
#
#     Returns:
#         model (nn.Module): The loaded segmentation model.
#     """
#     model = torch.load(model_path, map_location=torch.device('mps' if torch.cuda.is_available() else 'cpu'))
#     print(f'\nModel loaded from "{model_path}" at time {time.ctime()}')
#
#     model.eval()  # Set the model to evaluation mode
#     print(f'Model set to evaluation mode at time {time.ctime()}')
#     return model
#
#
# def image_to_tensor(img, trained_model):
#     """
#     Converts an input image to a tensor and makes a prediction using a trained model.
#
#     Args:
#         img: The input image to be converted.
#         trained_model: The trained model used for making predictions.
#
#     Returns:
#         output_labels_np: A numpy array representing the predicted class labels.
#     """
#
#     # Define image preprocessing steps (resize, normalize, convert to tensor)
#     preprocess = transforms.Compose([
#         transforms.Resize((704, 1280)),  # Resize to model input dimensions
#         transforms.ToTensor(),           # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
#     ])
#
#     # Load the input image
#
#     # Preprocess the image
#     input_tensor = preprocess(img)
#     input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
#
#     # Move the input tensor to the device (GPU or CPU)
#     device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
#     input_tensor = input_tensor.to(device)
#     trained_model = trained_model.to(device)
#
#     # Make the prediction
#     with torch.no_grad():
#         output = trained_model(input_tensor)
#
#     # The output is a tensor with shape [1, 23, 704, 1280]
#     # If you want to convert this to class labels (argmax across the channel dimension)
#     output_labels = torch.argmax(output, dim=1)
#
#     # Convert the tensor back to a numpy array (optional)
#     output_labels_np = output_labels.squeeze().cpu().numpy()
#
#     return output_labels_np

# public libraries
from torchvision import transforms
import torch
import time


def load_segmentation_model(model_path):
    """
    Load a trained segmentation model from a file.

    Args:
        model_path (str): Path to the model file (.pt).

    Returns:
        model (nn.Module): The loaded segmentation model.
    """
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.to(device)  # Move model to the device
    print(f'\nModel loaded from "{model_path}" at time {time.ctime()}')

    model.eval()  # Set the model to evaluation mode
    print(f'Model set to evaluation mode at time {time.ctime()}')
    return model, device  # Return the device as well


# def image_to_tensor(img, trained_model, device):
#     """
#     Converts an input image to a tensor and makes a prediction using a trained model.
#
#     Args:
#         img: The input image to be converted.
#         trained_model: The trained model used for making predictions.
#         device: The device to perform inference on (GPU or CPU).
#
#     Returns:
#         output_labels_np: A numpy array representing the predicted class labels.
#     """
#
#     # Define image preprocessing steps (resize, normalize, convert to tensor)
#     preprocess = transforms.Compose([
#         transforms.Resize((704, 1280)),  # Resize to model input dimensions
#         transforms.ToTensor(),           # Convert to tensor
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
#     ])
#
#     # Preprocess the image
#     input_tensor = preprocess(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
#
#     # Make the prediction
#     with torch.no_grad():
#         output = trained_model(input_tensor)
#
#     # Convert the output to class labels and then to a numpy array
#     output_labels_np = torch.argmax(output, dim=1).squeeze().cpu().numpy()
#
#     return output_labels_np

# Define preprocessing steps globally to avoid re-definition
preprocess = transforms.Compose([
    transforms.Resize((704, 1280)),  # Resize to model input dimensions
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def image_to_tensor(img, trained_model, device):
    """
    Converts an input image to a tensor and makes a prediction using a trained model.

    Args:
        img: The input image to be converted.
        trained_model: The trained model used for making predictions.
        device: The device to perform inference on (GPU or CPU).

    Returns:
        output_labels_np: A numpy array representing the predicted class labels.
    """

    # Preprocess the image
    input_tensor = preprocess(img).unsqueeze(0).to('cpu')  # Add batch dimension and move to device

    # Make the prediction
    with torch.no_grad():
        output = trained_model(input_tensor)

    # Convert the output to class labels and then to a numpy array
    output_labels_np = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    return output_labels_np
