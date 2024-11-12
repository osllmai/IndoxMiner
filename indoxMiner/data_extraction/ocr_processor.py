import cv2
import pandas as pd

class OCRProcessor:
    def __init__(self, model: str = 'tesseract'):
        self.model = model.lower()
        self.ocr = None

        if self.model == 'paddle':
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(lang='en')
            except ImportError:
                raise ImportError("Please install paddleocr package to use PaddleOCR")

        elif self.model == 'easyocr':
            try:
                import easyocr
                self.ocr = easyocr.Reader(['en'])
            except ImportError:
                raise ImportError("Please install easyocr package to use EasyOCR")

    def preprocess_image_for_tesseract(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary_image = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        return binary_image

    def extract_text_with_tesseract(self, image_path: str) -> str:
        try:
            import pytesseract
            processed_image = self.preprocess_image_for_tesseract(image_path)
            text = pytesseract.image_to_string(processed_image, config="--oem 3 --psm 6")
            return text.strip()
        except ImportError:
            raise ImportError("Please install pytesseract package to use Tesseract OCR")

    def preprocess_image_for_easyocr(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        return dilated_image

    def extract_text_with_easyocr(self, image_path: str) -> str:
        processed_image = self.preprocess_image_for_easyocr(image_path)
        results = self.ocr.readtext(processed_image, text_threshold=0.6, low_text=0.3)
        text_data = [text for (_, text, confidence) in results]
        # Join detected text lines into a single string
        return "\n".join(text_data)

    def extract_text_with_paddle(self, image_path: str) -> str:
        result = self.ocr.ocr(image_path, rec=True)
        text_lines = [line[1][0] for res in result for line in res if line[1][0].strip()]
        return "\n".join(text_lines)

    def extract_text(self, image_path: str):
        if self.model == 'tesseract':
            return self.extract_text_with_tesseract(image_path)
        elif self.model == 'paddle':
            return self.extract_text_with_paddle(image_path)
        elif self.model == 'easyocr':
            return self.extract_text_with_easyocr(image_path)
        else:
            raise ValueError("Invalid OCR model selected. Choose 'tesseract', 'paddle', or 'easyocr'.")
