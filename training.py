from ultralytics import YOLO
import torch

def train_model():
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the YOLOv8 segmentation model
    model = YOLO("yolov8n-seg.pt")  # Use segmentation model

    # Specify the path to your updated data.yaml file
    dataset_path = "D:/Kazim/Teeth Detection.v2i.yolov8/data.yaml"

    # Train the model
    results = model.train(
        data=dataset_path,
        epochs=100,
        batch=16,
        imgsz=640,
        device=device,
        name="plaque-segmentation"
    )

    # Evaluate the model on the validation set
    results = model.val()

    # Export the trained model
    model.export(format="onnx")

if __name__ == "__main__":
    train_model()