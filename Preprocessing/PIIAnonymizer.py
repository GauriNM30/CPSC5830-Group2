from transformers import pipeline
import torch

class PIIAnonymizer:
    def __init__(self):
        try:
            # Check if GPU is available and set device accordingly
            self.device = 0 if torch.cuda.is_available() else -1

            # Load the PII detection pipeline using the urchade/gliner_multi_pii-v1 model
            self.pii_detector = pipeline("ner", model="urchade/gliner_multi_pii-v1", device=self.device)
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            self.pii_detector = None
    
    def anonymize(self, input_text):
        if not self.pii_detector:
            print("Model not initialized properly.")
            return input_text
        
        try:
            # Detect PII entities in the input text
            detected_entities = self.pii_detector(input_text)

            # Replace detected PII with a generic [ANONYMIZED] tag
            anonymized_text = input_text
            for entity in detected_entities:
                entity_text = entity["word"]
                anonymized_text = anonymized_text.replace(entity_text, "[ANONYMIZED]")

            return anonymized_text
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return input_text  # Return original text if an error occurs

