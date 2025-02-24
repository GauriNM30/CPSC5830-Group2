import re
from gliner import GLiNER

class PIIAnonymizer:
    def __init__(self):
        """
        Initialize the PIIAnonymizer class by loading the GLiNER model.
        """
        self.model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")
        self.labels = [
            "person", "organization", "phone number", "address", "passport number", "email", 
            "credit card number", "social security number", "health insurance id number", 
            "date of birth", "mobile phone number", "bank account number", "medication", 
            "cpf", "driver's license number", "tax identification number", "medical condition", 
            "identity card number", "national id number", "ip address", "email address", 
            "iban", "credit card expiration date", "username", "health insurance number", 
            "registration number", "student id number", "insurance number", "flight number", 
            "landline phone number", "blood type", "cvv", "reservation number", 
            "digital signature", "social media handle", "license plate number", "cnpj", 
            "postal code", "passport_number", "serial number", "vehicle registration number", 
            "credit card brand", "fax number", "visa number", "insurance company", 
            "identity document number", "transaction number", "national health insurance number", 
            "cvc", "birth certificate number", "train ticket number", "passport expiration date", 
            "social_security_number"
        ]

    def detect_entities(self, text):
        """
        Detect PII entities in the input text using custom regex patterns and the GLiNER model.
        Custom labels are prioritized over generic labels.

        Args:
            text (str): The input text to analyze.

        Returns:
            list: A list of dictionaries containing detected entities and their labels.
        """
        entities = []

        # Step 1: Detect custom entities using regex (prioritized)
        custom_patterns = {
            "SEVIS_ID": r"\bN\d{10}\b",  # SEVIS ID regex: Starts with N followed by 10 digits
            "I94_NUMBER": r"\b\d{9}[A-Za-z]\d\b",  # I-94 regex: 9 digits, 1 letter, 1 digit
            "EMPLOYER_ID": r"\b\d{7}\b",  # Employer ID regex: 7 digits
        }

        for label, pattern in custom_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": label  # Custom label
                })

        # Step 2: Detect entities using the GLiNER model (generic labels)
        gliner_entities = self.model.predict_entities(text, self.labels)
        for entity in gliner_entities:
            # Skip if the entity text is already covered by a custom label
            if not any(entity["text"] in e["text"] for e in entities):
                entities.append(entity)

        return entities

    def anonymize_text(self, text):
        """
        Anonymize the input text by replacing detected PII entities with placeholders.

        Args:
            text (str): The input text to anonymize.

        Returns:
            str: The anonymized text.
        """
        entities = self.detect_entities(text)
        anonymized_text = text
        for entity in entities:
            anonymized_text = anonymized_text.replace(entity["text"], f"<{entity['label'].upper()}>")
        return anonymized_text


# # Example usage
# if __name__ == "__main__":
#     # Input text
#     text = """
#     Hi, my name is John Doe, SEVIS ID N1234567890, and my employer id is 4196688, and my passport is t47890990 and my email is john.doe@university.edu.
#     My phone number is +1-555-123-4567, and my SSN is 123-45-6789. I was born on 01/01/1995 in Springfield, Illinois, USA.
#     My driver's license is AB-1234567, and my VIN is 1HGCM82633A123456. My I-94 number is 123456789A1.
#     """

#     # Initialize the PIIAnonymizer
#     anonymizer = PIIAnonymizer()

#     # Anonymize the text
#     anonymized_text = anonymizer.anonymize_text(text)
#     # Print the anonymized text
#     print("Anonymized Text:")

#     print(anonymized_text)