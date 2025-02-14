# Import libraries
import google.generativeai as genai
import dotenv
import os
dotenv.load_dotenv()

class RailGuard:
    def __init__(self, model_name = 'gemini-2.0-flash-exp') -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        railguard_prompt = """
                           You are a railguard meant to judge if the user's question is related to international students studying in the United States on an F1 Visa. If the question is about F1 Visa rules, regulations, work authorization, or other instructions specifically for international students, reply 'True'; otherwise, reply 'False' regardless of whether the question is about other laws, rules, or regulations.
                            """
        self.chat_model = genai.GenerativeModel(model_name, system_instruction=railguard_prompt)

    def railguard_eval(self, question: str) -> bool:
        response = self.chat_model.generate_content(question)
        return response.text.strip().lower() == "true"