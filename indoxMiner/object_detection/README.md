# Object Detection Models - IndoxMiner

This repository contains various pre-trained object detection models integrated into the IndoxMiner framework. You can use these models to perform object detection on images by specifying the model and running the detection using an easy-to-use interface.

## Available Models

1. **RT-DETR**  
   A real-time object detection model based on the DEtection TRansformers (DETR) framework. It offers competitive performance and is optimized for speed.

2. **YOLOX**  
   A fast and efficient object detection model that builds upon the YOLO family. It delivers real-time detection with high accuracy.

3. **Grounding DINO**  
   A vision model that grounds objects in an image, combining transformers with dense prediction.

4. **YOLOv8**  
   The latest in the YOLO series, YOLOv8 continues the lineage of fast and accurate models for object detection with improvements in speed and accuracy.

5. **KOSMOS**  
   A versatile model for a variety of vision tasks, including object detection, with a focus on multi-modal capabilities.

6. **OWL-ViT**  
   A vision transformer model that leverages pre-trained knowledge for more accurate object detection. Suitable for a wide range of detection tasks.

7. **Detectron2**  
   Developed by Facebook AI Research (FAIR), Detectron2 is a modular framework for object detection and segmentation.

8. **SAM2**  
   A model that specializes in self-supervised learning, useful for detection with limited labeled data.

9. **LLAVANext**  
   An advanced object detection model that can handle a variety of input types, from images to video frames.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/osllmai/IndoxMiner.git
cd IndoxMiner
pip install -r requirements.txt
```

## Usage

Here is an example of how to use an object detection model with the IndoxMiner framework.

### Example for RT-DETR Model

1. **Set up the model**  
   Replace `"owlvit"` with the desired model name (e.g., `"rtdetr"`).

```python
from indoxminer.object_detection import IndoxObjectDetection

model_name = "rtdetr"  # Change to your desired model
image_path = "/content/download.jpg"  # Replace with your image path

# Create an instance of IndoxObjectDetection with the selected model
indox_detector = IndoxObjectDetection(model_name=model_name, device="cuda")

# Run the detection
indox_detector.run(image_path)
```

### Supported Models

You can select from the following models by changing the `model_name` parameter:

- `"rtdetr"`
- `"yolox"`
- `"groundingdino"`
- `"yolov8"`
- `"kosmos"`
- `"owlvit"`
- `"detectron2"`
- `"sam2"`
- `"llavanext"`

Each model may have different capabilities, and we are continuously adding more models to the repository.

## Additional Information

- For best performance, it's recommended to run these models on a machine with a GPU.
- You can customize the input queries (e.g., objects you want to detect) as required.
  
## Contributing

We welcome contributions! If you have suggestions for new models, improvements, or bug fixes, feel free to open an issue or submit a pull request.

---