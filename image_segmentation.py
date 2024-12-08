import os
import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from torchvision import models, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

# Step 1: Set environment variable to enable CuDNN-based attention kernels
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"

# Step 2: Define device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Build SAM2 model
sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(sam2)

# Step 4: Load ResNet50 with weights and move it to the device
from torchvision.models import ResNet50_Weights
classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
classifier.eval()
classifier = classifier.to(device)

# Step 5: Preprocessing pipeline for classifier
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 6: Function to classify a mask region
def classify_mask(mask, image):
    # Extract region of the image corresponding to the mask
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]  # Apply the mask to the image
    masked_image = Image.fromarray(masked_image)

    # Preprocess and pass through classifier
    input_tensor = preprocess(masked_image).unsqueeze(0).to(device)
    
    with autocast():  # Enable mixed precision inference
        with torch.no_grad():
            output = classifier(input_tensor)
            _, predicted_class = output.max(1)  # Get predicted class index
            
    return predicted_class.item()

# Step 7: Function to get PNG image names
def get_png_image_names(directory):
    return [f for f in os.listdir(directory) if f.endswith('.png')]

# Dataset and results directories
dataset_dir = "images/Unity"
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)  # Create results directory if it doesn't exist

# Get the list of png images in the dataset folder
png_images = get_png_image_names(dataset_dir)

for i, image_name in enumerate(png_images):
    try:
        # Step 8: Read the image
        image_number = image_name.split(".")[0]
        image_path = os.path.join(dataset_dir, image_name)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))

        # Step 9: Generate masks using the mask generator
        with torch.no_grad():  # Avoid storing intermediate values in computation graph
            masks2 = mask_generator.generate(image)

        # Step 10: Initialize the semantic segmentation map with int32 instead of uint8
        semantic_map = np.zeros(image.shape[:2], dtype=np.int32)

        # Step 11: Classify each mask and fill the semantic map
        for mask in masks2:
            mask_area = mask['segmentation']  # The binary mask
            predicted_class = classify_mask(mask_area, image)  # Classify the mask
            predicted_class = predicted_class % 256  # Optional: Limit class to 0-255 if needed
            semantic_map[mask_area] = predicted_class  # Assign class label to the semantic map

        # Step 12: Visualize or save the results
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(semantic_map)  # Show semantic map
        ax.axis('off')

        output_path = os.path.join(results_dir, f"segmented_{image_name}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
        plt.close(fig)

        # Step 13: Clear variables and free up GPU memory
        del image, masks2  # Delete references to free memory
        torch.cuda.empty_cache()  # Clear GPU cache to free memory

    except Exception as e:
        print(f"Error processing {image_name}: {e}")
