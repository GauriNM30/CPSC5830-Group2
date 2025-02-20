from railguard import railguard
#from pinecone import pinecone
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    #BitsAndBytesConfig,
    pipeline
) 
import dotenv
import loguru
import json
import os

class Capstone:
    def __init__(self)->None:

        self.logger = loguru.logger
        self.logger.add("logs/capstone.log", format="{time} {level} {message}", level="TRACE")

        dotenv.load_dotenv()
        self.railguard = railguard.RailGuard()

        # TODO: Get pinecone/pinecone.py working
        self.pinecone = pinecone.Pinecone()

        with open('config.json') as f:
            config = json.load(f)
        model_name = config['model_name']

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

        self.logger.success("Capstone Initialized Successfully")
        
    def format_model_question(self, question:str, pinecone_response:str)->str:
        # TODO: Add the correct prompt that combines question and the pinecone data
        return f"Question: {question} \Extra data for answer: {pinecone_response}"
        


    def chat(self, question:str)->str:
        retry = 0

        # Railguard call
        railguard_response = self.railguard.railguard_eval(question)
        
        if railguard_response:
            while retry < 3:
                try:
                    #TODO: Pinecone call
                    pinecone_response = self.pinecone.query_pinecone(question,top_k=5)
            
                    try:
                        # Format the question for the model
                        prompt = self.format_model_question(question, pinecone_response)

                        # Generate the response
                        response = self.generator(prompt, max_length=50, num_return_sequences=1)

                        self.logger.success(f"Question: '{question}', 
                                            Pinecone Response: '{pinecone_response}', 
                                            Answer: '{response[0]['generated_text']}' ")
                        
                        return response[0]['generated_text']
                    except Exception as e:
                        self.logger.error(f"Error in Generating Text: {e}")
                        retry += 1
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error in Pinecone: {e}")
                    retry += 1
                    continue
        else:
            return "Sorry, I am not able to answer your question, please ask question related to F1."