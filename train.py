import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import evaluate
from transformers import ViTImageProcessor
from transformers import EarlyStoppingCallback

train_dir = "./train" #Path to training dataset
val_dir = "./validation" #Path to validation dataset
test_dir = "./test" #Path to test dataset
data_dir = "./Trained Model" #Path to model folder
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k") #Retrieve pre-trained model from HuggingFace

#Function to process images
def transform(image):
    inputs = image_processor(image, return_tensors="pt")
    return inputs["pixel_values"].squeeze(0)

#Retrieve and map datasets based on folder name
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

print(train_dataset)

#Group data and measure training metrics
def collate_fn(batch):
    images , labels = zip(*batch)
    return{
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(labels)
    }

metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions = np.argmax(p.predictions, axis=1)
    references = p.label_ids
    return metric.compute(predictions=predictions, references=references)

num_classes = len(train_dataset.classes)

#Retrieve model and link to GPU/CPU
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

training_args = TrainingArguments(
    output_dir=data_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=10,
    fp16=True,
    logging_steps=10,
    learning_rate=2e-5,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=image_processor,
    callbacks=[early_stopping_callback],
)

train_results = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)


