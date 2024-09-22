# from transformers import AutoImageProcessor, ResNetModel
# import torch
# from datasets import load_dataset
# from torchvision import transforms
# import numpy as np
# import evaluate

# mnist = load_dataset("mnist")
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# def preprocess(example):
#     example['image'] = transform(example['image'])
#     return example

# mnist = mnist.map(preprocess)

# image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
# model = ResNetModel.from_pretrained("microsoft/resnet-50")

# inputs = image_processor(mnist["test"]["image"], return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     print(outputs)
    
# last_hidden_states = outputs.last_hidden_state
# list(last_hidden_states.shape)

import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, ResNetModel
from torch.utils.data import DataLoader
import numpy as np
import evaluate

# Load the pretrained ResNet-50 model and image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetModel.from_pretrained("microsoft/resnet-50")

model.eval()  # Set model to evaluation mode

# Load MNIST dataset
mnist = load_dataset('mnist')

# Function to preprocess MNIST images to fit ResNet input size (224x224, 3 channels)
def preprocess_image(image):
    image = image.convert("RGB")  # Convert to RGB
    return image_processor(image, return_tensors="pt").pixel_values.squeeze(0)  # Preprocess image to tensor
# Since we're applying preprocessing one image at a time in collate_fn, we need to remove excess dimensions with .squeeze(0)

def collate_fn(batch):
    images = [preprocess_image(item['image']) for item in batch]  # Preprocess each image
    labels = torch.tensor([item['label'] for item in batch])  # Extract labels as tensors
    return torch.stack(images), labels  # Return images as a stacked tensor and labels

# Create DataLoader for batching and processing
test_loader = DataLoader(mnist['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)

correct_predictions = 0
total_predictions = 0

# Run inference over the test dataset
with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)

        logits = outputs.pooler_output  # Get the final feature embeddings from the model's pooler
        predicted_labels = torch.argmax(logits, dim=1)

        # Calculate accuracy
        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

# Compute overall accuracy
accuracy = correct_predictions / total_predictions * 100
print(f"Accuracy on MNIST dataset: {accuracy:.2f}%")


