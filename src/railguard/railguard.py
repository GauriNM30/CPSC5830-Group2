# Import libraries
import google.generativeai as genai
import dotenv
import os
dotenv.load_dotenv()

class RailGuard:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = 'gemini-2.0-flash-exp'
        genai.configure(api_key=api_key)
        railguard_prompt = """
                            You are a railguard meant to judge if the user question is related to Law or not. Particularly if it is related to rules or regulations or if the question asks anything related to laws or rules or instruction related to international students studying in the United States on an F1 Visa. 
                            If the question is related, reply True; else, reply False.
                            """
        self.chat_model = genai.GenerativeModel(model_name, system_instruction=railguard_prompt)

    def railguard_eval(self, question: str) -> bool:
        response = self.chat_model.generate_content(question)
        return response.text.strip().lower() == "true"
