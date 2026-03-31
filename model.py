import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, height, width, channel, nclasses):
        super().__init__()
        
        # Load VGG16
        vgg = models.vgg16(pretrained=True)
        
        # Freeze VGG16 layers
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Remove the classifier part of VGG16
        self.features = vgg.features

        # Flatten layer
        self.flatten = nn.Flatten() 
        
        # Label branch
        self.class_branch = nn.Sequential(
            nn.Linear(512 * (height // 32) * (width // 32), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, nclasses),
            nn.Softmax(dim=1)
        )
        
        # Bounding box branch
        self.bbox_branch = nn.Sequential(
            nn.Linear(512 * (height // 32) * (width // 32), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        
        label = self.class_branch(x)
        bbox = self.bbox_branch(x)
        
        return label, bbox

# Example usage
if __name__ == "__main__":
    height, width, channel, nclasses = 224, 224, 3, 5
    model = MyModel(height, width, channel, nclasses)
    print(model)

    # Example input
    input_tensor = torch.randn(1, channel, height, width)
    label, bbox = model(input_tensor)
    print(label.shape, bbox.shape)