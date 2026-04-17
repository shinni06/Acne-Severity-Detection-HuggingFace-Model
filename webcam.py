import cv2
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

model_id = "shinni06/Acne_Severity_Detection"
model = ViTForImageClassification.from_pretrained(model_id)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
image_processor = ViTImageProcessor.from_pretrained(model_id)

class_names = ['clear_skin', 'mild_acne', 'severe_acne'] 

#Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

window_name = "Doping Symptom Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    inputs = image_processor(pil_image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        label = class_names[prediction]

    #Create colour indicator
    if label == 'severe_acne':
        indicator = (0, 0, 255) #Red - Severe Acne
    elif label == 'mild_acne':
        indicator = (0, 255, 255) #Yellow - Mild Acne
    else:
        indicator = (0, 255, 0) #Green - Clear Skin

    cv2.circle(frame, (35, 32), 10, indicator, -1)

    #Display detected skin condition
    cv2.putText(frame, f"Condition: {label}", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow(window_name, frame)

    cv2.waitKey(1)

    #Close the process if the X button on the window is clicked
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()