import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai
import time

# Updated RailGuard class to handle previous conversation context
class RailGuard:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = 'gemini-2.0-flash-exp'
        genai.configure(api_key=api_key)
        railguard_prompt = f"""
You are a classification tool that evaluates whether user questions relate to F-1 visa regulations, CPT, OPT, or follow-up conversations about these topics. Analyze the query and respond with **True** or **False** based on the following:

1. **True** - Respond when the query directly or thematically relates to:
   - F-1 visa rules, study requirements, work authorization (CPT/OPT), SEVIS, duration of status, or related compliance matters for U.S. international students.
   - Implied connections to employment eligibility, visa maintenance, curricular training, or post-completion practical training.
   - Follow-up or continuation of previous conversations about F-1 visas, CPT, OPT, or similar topics (e.g., "also," "another question," or "following up").

2. **False** - Respond when the query is completely unrelated to U.S. student visas, including:
   - Topics concerning tourist visas, unrelated immigration matters, or non-visa-related subjects.
   - Any questions that do not touch on F-1 visas, CPT, OPT, or related regulations or concerns.
   - Example, any questions related to Green Card, H1B, O1 Visa, strictly reply False.
"""
        self.chat_model = genai.GenerativeModel(model_name, system_instruction=railguard_prompt)

    def railguard_eval(self, question: str, previous_context: str = "") -> bool:
        if previous_context:
            input_text = f"Previous conversation: {previous_context}\nCurrent question: {question}"
        else:
            input_text = question

        response = self.chat_model.generate_content(input_text)
        time.sleep(6)  # Delay of 6 seconds after the Gemini model call
        return response.text.strip().lower() == "true"

# Updated RAG class that uses Pinecone to enrich the first prompt with rich context
class RAG:
    def __init__(self):
        # Initialize the device, model pipeline, Pinecone index, embedding model, and RailGuard
        self.device = self.get_device()
        self.pipe = self.initialize_model()
        self.index = self.initialize_pinecone()
        self.embedding_model = self.initialize_embedding_model()
        self.reasoning_prompt = self.load_reasoning_prompt()  # Cache prompt from file
        self.railguard = RailGuard()  # Initialize RailGuard
        self.context = ""

    def get_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()  # Clear any unused GPU memory
                return torch.device("cuda")
            except RuntimeError:
                print("GPU out of memory. Falling back to CPU.")
        return torch.device("cpu")

    def set_context(self, context_fetched):
        self.context = context_fetched
        
    def get_context(self):
        return self.context
    
    def initialize_model(self):
        try:
            if self.device.type == "cuda":
                # Load model in FP16 to reduce GPU memory usage
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2-7B-Instruct",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                # Use CPU if GPU is not available
                pipe = pipeline(
                    "text-generation",
                    model="Qwen/Qwen2-7B-Instruct",
                    device=-1
                )
            print(f"Using device: {self.device}")
            return pipe
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU ran out of memory. Clearing memory and retrying on CPU...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                self.device = torch.device("cpu")
                return pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct", device=-1)
            else:
                raise e

    def get_api_key(self, api_name):
        env_path = "../.dummy_env"  # Adjust the path as needed
        load_dotenv(env_path)
        return os.getenv(api_name)

    def initialize_pinecone(self):
        pinecone_api_key = self.get_api_key("PINECONE_API_KEY")
        pinecone = Pinecone(api_key=pinecone_api_key)
        index_name = "recursive-chunks"
        index = pinecone.Index(index_name)
        return index

    def initialize_embedding_model(self):
        # Run embedding model on CPU to preserve GPU memory
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def load_reasoning_prompt(self):
        try:
            with open("prompt.txt", "r") as f:
                return f.read().strip()
        except Exception as e:
            print("Error loading reasoning prompt:", e)
            return ""

    def query_pinecone(self, query_text, top_k=10):
        if not hasattr(self, 'embedding_model') or not hasattr(self, 'index'):
            raise ValueError("Embedding model and index must be defined before running the function.")

        query_embedding = self.embedding_model.encode(query_text).tolist()
        return self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    def generate_answer(self, query_text, previous_context: str = ""):
        # Check if the question is relevant using RailGuard, taking previous context into account
        is_relevant = self.railguard.railguard_eval(query_text, previous_context)
        if not is_relevant:
            return "I specialize in F-1 visa regulations, CPT, and OPT. Please ask related questions."

        # Query Pinecone for rich context
        results = self.query_pinecone(query_text)
        context = []
        for match in results.get("matches", []):
            if "metadata" in match and "text" in match["metadata"]:
                context.append(match["metadata"]["text"])
            else:
                print(f"Skipping invalid match: {match}")
        rich_context = "\n".join(context)
        #set context variable in class (for evaluation)
        self.set_context(rich_context)
        # Construct the prompt differently depending on whether it's the first prompt or a follow-up
        if not previous_context:
            # First prompt: include only rich context from Pinecone
            prompt = (
                f"<Question>: {query_text}\n </Question>\n\n"
                f"{self.reasoning_prompt}\n\n"
                f"<context> \n Use this rich context to answer if needed:\n{rich_context}\n</context>\n\n"
                "Re evaluate your response again finally before providing answer."
            )
        else:
            # Follow-up: include previous conversation context plus new rich context
            prompt = (
                f"<Question>: {query_text}\n </Question>\n\n"
                f"{self.reasoning_prompt}\n\n"
                f"<context>:\n{previous_context}\n</context>\n "
                f"<more_context> \n Use this rich context to answer if needed:\n{rich_context}\n</more_context>\n"
                "Re evaluate your response again finally before providing answer."
            )

        # Generate answer using no_grad to avoid unnecessary computations
        with torch.no_grad():
            generated = self.pipe(
                prompt,
                max_new_tokens=250,
                num_return_sequences=1,
                temperature=0.5,
                top_p=0.9
            )

        full_generated_text = generated[0]['generated_text']
        final_answer = full_generated_text[len(prompt):].strip()

        # Optionally clear GPU cache after generation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return final_answer