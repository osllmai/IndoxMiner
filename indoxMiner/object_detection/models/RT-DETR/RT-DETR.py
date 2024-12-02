import torch
import requests
from PIL import Image
import supervision as sv
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import matplotlib.pyplot as plt

class RTDETRModel:
    def __init__(self, checkpoint="PekingU/rtdetr_r50vd_coco_o365", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)

    def detect_objects(self, image_url, threshold=0.1):  # Lowered threshold
        # Load image from URL
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get results
        w, h = image.size
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=[(h, w)], threshold=threshold
        )
        return image, results[0]



    def visualize_results(self, image, results):
        # Convert detections from the model's output
        detections = sv.Detections.from_transformers(results)

        # Generate labels for the detections
        labels = [
            f"{self.model.config.id2label[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate the image
        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)

        # Add labels to the annotated image
        label_annotator = sv.LabelAnnotator()
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Save the annotated image
        annotated_image.save("annotated_image.jpg")

        # Display the annotated image
        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title("Annotated Image")
        plt.show()

        return annotated_image


