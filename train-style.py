import data
from model import MyModel
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Kích thước dữ liệu
    height, width, channel, nclasses = 224, 224, 3, 5

    # Load data
    images, labels, bboxes = data.get_data(height=height, width=width)

    train_set, val_set = data.preprocess_input(images, labels, bboxes)
    images_train, labels_train, bboxes_train = train_set
    images_valid, labels_valid, bboxes_valid = val_set

    # Convert data to PyTorch tensors
    images_train = torch.tensor(np.stack(images_train)).permute(0, 3, 1, 2).float()
    labels_train = torch.tensor(np.stack(labels_train)).float()
    bboxes_train = torch.tensor(np.stack(bboxes_train)).float()

    images_valid = torch.tensor(np.stack(images_valid)).permute(0, 3, 1, 2).float()
    labels_valid = torch.tensor(np.stack(labels_valid)).float()
    bboxes_valid = torch.tensor(np.stack(bboxes_valid)).float()

    # Create DataLoader
    train_dataset = TensorDataset(images_train, labels_train, bboxes_train)
    val_dataset = TensorDataset(images_valid, labels_valid, bboxes_valid)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MyModel(height, width, channel, nclasses)
    model = model.to(device)

    # Define loss functions and optimizer
    criterion_label = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Initialize best validation loss
    val_best = float('inf')

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels, bboxes in train_loader:
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            optimizer.zero_grad()
            outputs_label, outputs_bbox = model(images)
            loss_label = criterion_label(outputs_label, labels)
            loss_bbox = criterion_bbox(outputs_bbox, bboxes)
            loss_label_weight = 1.0
            loss_bbox_weight = 1.0
            loss = loss_label_weight * loss_label + loss_bbox * loss_bbox_weight 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for images, labels, bboxes in val_loader:
                images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
                outputs_label, outputs_bbox = model(images)
                loss_label = criterion_label(outputs_label, labels)
                loss_bbox = criterion_bbox(outputs_bbox, bboxes)
                loss = loss_label + loss_bbox
                val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss}")

            # Save the model if validation loss is lower than the best validation loss
            if val_loss < val_best:
                val_best = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                print("Model saved with validation loss: {:.4f}".format(val_loss))