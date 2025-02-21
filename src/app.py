from railguard import railguard
import dotenv
import loguru
import json
import os
import google.generativeai as genai
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Pinecone
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pinecone_code.pinecone_code import Pinecone_code

class Capstone:
    def __init__(self)->None:
        self.logger = loguru.logger
        self.logger.add("logs/capstone.log", format="{time} {level} {message}", level="TRACE")

        dotenv.load_dotenv()
        self.railguard = railguard.RailGuard()

        # Initialize Pinecone
        self.pinecone = Pinecone_code()

        # Initialize Langchain components with streaming
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo-16k",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Initialize the conversational chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.pinecone.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True
        )

        # Initialize Gemini as fallback
        api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyCa6CV8idLGFpzEGWZiEJYkiFU7JlCypi4')
        model_name = 'gemini-2.0-flash-exp'
        genai.configure(api_key=api_key)
        railguard_prompt = """
            You are a LAW model, you will be given related information and a question, please answer the question.
            """
        self.chat_model = genai.GenerativeModel(system_instruction=railguard_prompt)

        self.logger.success("Capstone Initialized Successfully")
        
    def format_model_question(self, question: str, pinecone_response: list[str]) -> str:
        additional_data = "\n".join(pinecone_response)
        return f"\nQuestion: {question} \nExtra data for answer: \n\n{additional_data}"
        
    def clean_pinecone_response(self, pinecone_response):
        return [match['metadata']['text'] for match in pinecone_response.get('matches', [])]

    def get_sources(self, response):
        if not response.get('source_documents'):
            return []
        
        sources = []
        for doc in response['source_documents']:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.append(doc.metadata['source'])
        return list(set(sources))

    def chat(self, question: str) -> dict:
        try:
            # Get context from Pinecone
            pinecone_response = self.pinecone.query_pinecone(question)
            cleaned_response = self.clean_pinecone_response(pinecone_response)
            
            # Use Railguard for safety check
            railguard_response = self.railguard.check_question(question)
            if not railguard_response:
                return {
                    "answer": "I apologize, but I cannot assist with that type of question.",
                    "sources": []
                }

            # Use Langchain for complex queries and context-aware responses
            langchain_response = self.conversation_chain({"question": question})
            answer = langchain_response['answer']
            sources = self.get_sources(langchain_response)

            # If Langchain doesn't provide a good response, fall back to Gemini
            if not answer or len(answer.strip()) < 10:
                formatted_question = self.format_model_question(question, cleaned_response)
                gemini_response = self.chat_model.generate_content(formatted_question)
                answer = gemini_response.text

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            self.logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "sources": []
            }

if __name__ == '__main__':
    capstone = Capstone()
    response = capstone.chat("What is the process for F1 visa application?")
    print(f"Answer: {response['answer']}")
    print(f"Sources: {response['sources']}")