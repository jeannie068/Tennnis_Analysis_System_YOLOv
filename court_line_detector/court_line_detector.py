import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models
from .tracknet import TrackerNet
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2) # Replace the last fully connected layer, 14 key point with x,y
        # self.model = TrackerNet(out_channels=15)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval() # 進入評估狀態 
        # Standardize and normalize images
        self.transform = transforms.Compose([
                         transforms.ToPILImage(),
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    
    # Predict the first frame only
    def predict(self, image):    
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0) # Turn one frame to a list for predicting
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        
        # Adjust x and y coordinates
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    # Draw Key Points on output video
    def draw_keypoints(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            # Plot keypoints on the image
            for i in range(0, len(keypoints), 2):
                x = int(keypoints[i])
                y = int(keypoints[i+1])
                cv2.putText(frame, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Key Points number
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1) # Key Points
            output_video_frames.append(frame)
        return output_video_frames
        