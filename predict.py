import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
from model import MyModel

def predict():
    # Load model
    model_path = './best_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel(224, 224, 3, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load image
    image_path = './dataset/images/octopus/image_0009.jpg'
    target_size = (224, 224)
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_img = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs_label, outputs_bbox = model(input_img)
        outputs_label = F.softmax(outputs_label, dim=1)
        label = torch.argmax(outputs_label, dim=1).item()
        bbox = outputs_bbox.squeeze().cpu().numpy()

    class_names = ['accordion', 'ant', 'buddha', 'camera','octopus']

    # Print predicted class and bounding box
    print(f"Predicted label: {class_names[label]}")
    print(f"Predicted bounding box: {bbox}")

    # Convert bbox from [0, 1] range to actual image dimensions
    original_width, original_height = image.size
    x_min = int(bbox[0] * original_width)
    y_min = int(bbox[1] * original_height)
    x_max = int(bbox[2] * original_width)
    y_max = int(bbox[3] * original_height)

    # Draw bounding box and label on the image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, class_names[label], (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the image
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    predict()