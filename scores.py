from ultralytics import YOLO

def evaluate_model():
    model_path = "runs/segment/plaque-segmentation3/weights/best.pt"
    dataset_path = "D:/Kazim/Teeth Detection.v2i.yolov8/data.yaml"

    # Load the trained model
    model = YOLO(model_path)

    # Run validation
    results = model.val(data=dataset_path)

    # Print model evaluation metrics
    print(f"Mean Average Precision (mAP@50): {results.box.map50:.3f}")
    print(f"Mean Average Precision (mAP@50-95): {results.box.map:.3f}")
    print(f"Precision: {results.box.precision:.3f}")
    print(f"Recall: {results.box.recall:.3f}")

if __name__ == '__main__':
    evaluate_model()
