"""
This script contains the code for the model inference.

Video frames are passed to the model and the predicted RGB mask is returned.
"""

from PIL import Image
from torchvision import transforms

img = Image.open('/kaggle/working/resized_original_images/041.png').convert('RGB')

# Ensure the model is in evaluation mode
model.eval()

# Load the trained model weights (if available)
# model.load_state_dict(torch.load('path_to_saved_model.pth'))

# Define image preprocessing steps (resize, normalize, convert to tensor)
preprocess = transforms.Compose([
    transforms.Resize((704, 1280)),  # Resize to model input dimensions
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet mean and std
])

# Load the input image

# Preprocess the image
input_tensor = preprocess(img)
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

# Move the input tensor to the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
model = model.to(device)

# Make the prediction
with torch.no_grad():
    output = model(input_tensor)

# The output is a tensor with shape [1, 23, 704, 1280]
# If you want to convert this to class labels (argmax across the channel dimension)
output_labels = torch.argmax(output, dim=1)

# Convert the tensor back to a numpy array (optional)
output_labels_np = output_labels.squeeze().cpu().numpy()

# You can visualize the output_labels_np or save it as an image using PIL or other libraries
