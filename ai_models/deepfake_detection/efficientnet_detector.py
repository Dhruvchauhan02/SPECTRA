import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import cv2


class EfficientNetDeepfakeDetector:
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        print("Loading model from:", model_path)

        # Load architecture
        self.model = EfficientNet.from_name('efficientnet-b0')

        # Replace classifier
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, 1)

        # Load YOUR trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def predict_proba(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            logit = output.item()
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1 - prob_real
            print("Logit:", logit)
            print("Prob real:", prob_real)
            print("Prob fake:", prob_fake)

        # 🔥 IMPORTANT: because fake=0, real=1 during training
        p_fake = 1 - prob_real

        return float(p_fake)
