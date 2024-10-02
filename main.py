from ultralytics import YOLO
import os
import cv2
import torch

# Main function
if __name__ == "__main__":
    # Load the YOLOv8 model with pre-trained weights (or your fine-tuned weights)
    model = YOLO('yolov8n.pt')  # Update with the actual path to your weights

    # Specify the dataset paths for training and validation
    train_path = 'data/images/train'
    valid_path = 'data/images/valid'
    test_path = 'data/images/test'
    
    # Train the model
    model.train(
        data="Chess Pieces.yolov8-obb/data.yaml",  # Update with your data.yaml path
        epochs=50,  # Number of training epochs
        imgsz=416,  # Image size for training
        batch=32,   # Batch size
        project="chess_project",  # Folder to save the trained model
        name="Chess_Model_2"  # Model name
    )
