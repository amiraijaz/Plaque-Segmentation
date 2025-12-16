# Plaque-Segmentation

A deep learning-based plaque segmentation model trained using YOLOv8 for accurate detection and segmentation of dental plaque.

## 📊 Dataset

This project utilizes the **Universe Roboflow** dataset for training and validation. The dataset contains annotated images of dental plaque for object detection and segmentation tasks.

- **Source**: [Universe Roboflow](https://universe.roboflow.com/)
- **Task**: Instance Segmentation
- **Domain**: Dental/Medical Imaging

## 🚀 Training

The model was trained using **GPU acceleration** to optimize training time and performance.

### Training Configuration
- **Framework**: YOLOv8 (Ultralytics)
- **Hardware**: GPU-enabled training
- **Task**: Instance Segmentation

## 📈 Model Performance

The trained model achieved the following metrics on the validation set:

| Metric | Score |
|--------|-------|
| **mAP@50** | **0.805** (80.5%) |
| **mAP@50-95** | **0.680** (68.0%) |

### Metrics Explanation
- **mAP@50**: Mean Average Precision at IoU threshold of 0.50
- **mAP@50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/amiraijaz/Plaque-Segmentation.git
cd Plaque-Segmentation

# Install dependencies
pip install ultralytics opencv-python numpy
```

## 💻 Usage

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('path/to/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

## 📁 Project Structure

```
Plaque-Segmentation/
├── data/              # Dataset files
├── runs/              # Training runs and results
├── models/            # Trained model weights
└── README.md          # Project documentation
```

## 🎯 Results

The model demonstrates strong performance with:
- High precision in detecting plaque regions
- Robust segmentation accuracy across varying image conditions
- Efficient inference suitable for real-time applications

## 🔮 Future Work

- [ ] Improve model performance with data augmentation
- [ ] Experiment with different YOLOv8 model sizes
- [ ] Deploy model as a web application
- [ ] Expand dataset with more diverse samples

## 📝 License

This project is open-source and available under the MIT License.

## 👤 Author

**Amir Aijaz**
- GitHub: [@amiraijaz](https://github.com/amiraijaz)

## 🙏 Acknowledgments

- Universe Roboflow for providing the dataset
- Ultralytics for the YOLOv8 framework