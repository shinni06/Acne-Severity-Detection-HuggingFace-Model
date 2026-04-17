import os
import random
import torch
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification
import evaluate
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor

model_dir = "./Trained Model"
test_dir = "./test"

#Retrieve model
model = ViTForImageClassification.from_pretrained(model_dir)
model.eval() #Switch model to evaluation mode

#Link model to GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

#Retrieve and map test dataset based on folder name
test_dataset = ImageFolder(test_dir)
class_names = test_dataset.classes

all_images = [
    (os.path.join(root, file), os.path.basename(root)) #Retrieve path to image and image category based on folder name
    for root, _, files in os.walk(test_dir) #Go through all files in test dataset folder
    for file in files if file.endswith((".png", ".jpg", ".jpeg")) #Ensure only files with valid image format are retrieved
]

#Randomly retrieve 6 images from test dataset
random_image_paths = random.sample(all_images, 6)

#Display result
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()

for idx, (image_path, true_label) in enumerate(random_image_paths):
    sample_image = Image.open(image_path).convert("RGB")
    processed_sample = image_processor(sample_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**processed_sample)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = class_names[predicted_class]

    axes[idx].imshow(sample_image) 
    axes[idx].axis("off")
    axes[idx].set_title(f"True: {true_label}\nPrediction: {predicted_label}", fontsize = 12)
    
plt.tight_layout()
plt.show()