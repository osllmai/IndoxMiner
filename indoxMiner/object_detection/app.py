import torch
from PIL import Image
import os

# Import specific models
from object_detection.models.GroundingDINO.Groundingdino import GroundingDINOObjectDetector
from object_detection.models.Kosmos-2.Kosmos2 import Kosmos2ObjectDetector
from object_detection.models.llava-next.LLaVANext import LLaVANextObjectDetector
from object_detection.models.OWL-ViT.OWL-ViT import OWLVitModel
from object_detection.models.RT-DETR.RTDETR import RTDETRModel
from object_detection.models.YOLOX.YOLOX import YOLOXModel

class IndoxObjectDetector:
    def __init__(self, model_name, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # Load the selected model based on model_name
        if model_name == "Kosmos2":
            self.model = Kosmos2ObjectDetector(device=device)
        elif model_name == "RTDETR":
            self.model = RTDETRModel(device=device)
        elif model_name == "LLaVANext":
            self.model = LLaVANextObjectDetector(device=device)
        elif model_name == "GroundingDINO":
            self.model = GroundingDINOObjectDetector(config_path="path/to/config", model_weights_path="path/to/weights", device=device)
        elif model_name == "YOLOX":
            self.model = YOLOXModel(exp_file="path/to/exp", model_path="path/to/model", device=device)
        elif model_name == "OWLVit":
            self.model = OWLVitModel()
        else:
            raise ValueError(f"Model {model_name} not recognized!")

    def detect_objects(self, image_path, **kwargs):
        """
        Detect objects using the selected model.
        Args:
            image_path (str): Path to the image file.
            kwargs: Additional arguments for model-specific methods (e.g., queries for OWLVit).
        Returns:
            objects: Detected objects and their bounding boxes.
        """
        return self.model.detect_objects(image_path, **kwargs)

    def visualize_results(self, image, objects, **kwargs):
        """
        Visualize the detection results.
        Args:
            image (PIL.Image or numpy.ndarray): The image with detected objects.
            objects: The detected objects and bounding boxes.
            kwargs: Additional arguments for model-specific visualization (e.g., text for OWLVit).
        """
        return self.model.visualize_results(image, objects, **kwargs)

    def run(self, image_path, **kwargs):
        """
        Run the detection and visualization for the selected model.
        Args:
            image_path (str): Path to the image file.
            kwargs: Additional arguments for model-specific methods (e.g., queries for OWLVit).
        """
        # Detect objects
        objects = self.detect_objects(image_path, **kwargs)

        # Open image for visualization
        image = Image.open(image_path)

        # Visualize results
        self.visualize_results(image, objects, **kwargs)
