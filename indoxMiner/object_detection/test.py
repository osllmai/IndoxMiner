from .models.owlvit import OWLVitModel
from .models.yolox import YOLOXModel
from .models.rtdetr import RTDETRModel
from .models.kosmos2 import Kosmos2ObjectDetector  # Assuming Kosmos2Model is in models.kosmos2

def main():
    model_type = input("Enter model type (owlvit/yolox/rtdetr/kosmos2): ").strip().lower()

    if model_type == "owlvit":
        # OWL-ViT Example
        detector = OWLVitModel()
        image_path = "/content/download.jpg"  # Replace with your image path
        queries = ["a cat", "a dog", "a person", "a car", "a tree"]
        image, results = detector.detect_objects(image_path, queries)
        detector.visualize_results(image, results, queries)

    elif model_type == "yolox":
        # YOLOX Example
        exp_file = "/content/YOLOX/exps/default/yolox_s.py"  # Path to YOLOX experiment file
        model_path = "/content/yolox_s.pth"  # Path to YOLOX weights
        image_path = "/content/download.jpg"  # Replace with your image path

        detector = YOLOXModel(exp_file, model_path)
        image, outputs, scale = detector.detect_objects(image_path)
        detector.visualize_results(image, outputs, scale)

    elif model_type == "rtdetr":
        detector = RTDETRModel()
        image_url = "https://media.istockphoto.com/id/627966690/photo/two-dogs-in-the-city.jpg?s=612x612&w=0&k=20&c=6Fj5qtEH9vs7ojnyfjF1mOgEA_i63rzAQtjtuVuw37A="  # Replace with your image URL
        image, results = detector.detect_objects(image_url)
        annotated_image = detector.visualize_results(image, results)

        # Display the annotated image
        annotated_image.thumbnail((600, 600))
        annotated_image.show()

    elif model_type == "kosmos2":
        detector = Kosmos2ObjectDetector()
        filename = '/content/download.jpg'  # Update the image path accordingly

        # Load the image using OpenCV
        image = cv2.imread(filename)

        # Check if the image is loaded successfully
        if image is None:
            print(f"Error: Unable to load image at {filename}")
            sys.exit(1)

        # Detect objects in the image
        objects = detector.detect_objects(image)

        # Visualize results
        annotated_image = detector.visualize_results(image, objects)

    else:
        print("Invalid model type. Please choose 'owlvit', 'yolox', 'rtdetr', or 'kosmos2'.")



if __name__ == "__main__":
    main()
