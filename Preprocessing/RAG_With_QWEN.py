import os
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# RailGuard class from railguard.py
class RailGuard:
    def __init__(self) -> None:
        api_key = os.getenv("GOOGLE_API_KEY")
        model_name = 'gemini-2.0-flash-exp'
        genai.configure(api_key=api_key)
        railguard_prompt = """
                            You are a railguard meant to judge if the user's question is related to F-1 visa regulations, CPT, or OPT. Specifically, determine if the question pertains to rules, regulations, or instructions for international students studying in the United States on an F-1 visa.

If the question is related, reply True; otherwise, reply False. 
                            """
        self.chat_model = genai.GenerativeModel(model_name, system_instruction=railguard_prompt)

    def railguard_eval(self, question: str) -> bool:
        response = self.chat_model.generate_content(question)
        return response.text.strip().lower() == "true"


# Updated RAG class
class RAG:
    def __init__(self):
        # Initialize the device, model pipeline, Pinecone index, embedding model, and RailGuard
        self.device = self.get_device()
        self.pipe = self.initialize_model()
        self.index = self.initialize_pinecone()
        self.embedding_model = self.initialize_embedding_model()
        self.reasoning_prompt = self.load_reasoning_prompt()  # Cache prompt from file
        self.railguard = RailGuard()  # Initialize RailGuard

    def get_device(self):
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()  # Clear any unused GPU memory
                return torch.device("cuda")
            except RuntimeError:
                print("GPU out of memory. Falling back to CPU.")
        return torch.device("cpu")

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
        index_name = "recursive-text-chunks-new"
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

    def query_pinecone(self, query_text, top_k=5):
        if not hasattr(self, 'embedding_model') or not hasattr(self, 'index'):
            raise ValueError("Embedding model and index must be defined before running the function.")

        query_embedding = self.embedding_model.encode(query_text).tolist()
        return self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    def generate_answer(self, query_text):
        # First, check if the question is relevant using RailGuard
        is_relevant = self.railguard.railguard_eval(query_text)
        if not is_relevant:
            return "This question is outside the scope of F-1 visa regulations, CPT, or OPT."

        # Query Pinecone for context
        results = self.query_pinecone(query_text)
        context = []
        for match in results.get("matches", []):
            if "metadata" in match and "text" in match["metadata"]:
                context.append(match["metadata"]["text"])
            else:
                print(f"Skipping invalid match: {match}")

        context_str = "\n".join(context)
        # Construct the prompt using the cached reasoning prompt
        prompt = (
            "Context:\n"
            f"{context_str}\n\n"
            f"Question: {query_text}\n\n"
            f"{self.reasoning_prompt}\n\n"
            "Answer concisely in 2-3 sentences if in scope, followed by 1 or 2 follow-up questions if needed. Don't answer for out of scope queries"
        )

        # Generate answer using no_grad to avoid unnecessary computations
        with torch.no_grad():
            generated = self.pipe(
                prompt,
                max_new_tokens=500,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9
            )

        full_generated_text = generated[0]['generated_text']
        final_answer = full_generated_text[len(prompt):].strip()

        # Optionally clear GPU cache after generation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return final_answer

    # def generate_answer(self, query_text):
    #     # Query Pinecone for context
    #     results = self.query_pinecone(query_text)
        
    #     # Safely extract and filter context
    #     context = []
    #     for match in results.get("matches", []):
    #         if "metadata" in match:
    #             metadata = match["metadata"]
    #             if "text" in metadata: # and match["score"] > 0.5:  # Filter by confidence score
    #                 context.append(metadata["text"])
    #             # else:
    #             #     print(f"Skipping low-confidence match: {match['id']}")
    #         else:
    #             print(f"Skipping match with missing metadata: {match['id']}")

    #     if not context:
    #         return "No relevant information found. Could you clarify your question or provide more details?"

    #     context_str = "\n".join(context)

    #     # Construct the prompt
    #     prompt = (
    #         f"{self.reasoning_prompt}\n\n"
    #         "Context:\n"
    #         f"{context_str}\n\n"
    #         f"Question: {query_text}\n"
    #         "Answer concisely in 2-3 sentences, followed by 1 or 2 follow-up questions."
    #     )

    #     # Generate answer
    #     with torch.no_grad():
    #         generated = self.pipe(
    #             prompt,
    #             max_new_tokens=500,
    #             num_return_sequences=1,
    #             temperature=0.7,
    #             top_p=0.9
    #         )
            
    #     # Extract the final answer
    #     full_generated_text = generated[0]['generated_text']
    #     final_answer = full_generated_text[len(prompt):].strip()

    #     # Optionally clear GPU cache after generation
    #     if self.device.type == "cuda":
    #         torch.cuda.empty_cache()

    #     return final_answer




























# import os
# import torch
# from transformers import pipeline
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone, ServerlessSpec
# # from PIIAnonymizer import PIIAnonymizer

# # Class to query pinecone and answer generation
# class RAG:
#     def __init__(self):
#         # Initialize the device and model pipeline
#         self.device = self.get_device()
#         self.pipe = self.initialize_model()
#         self.index = self.initialize_pinecone()
#         self.embedding_model = self.initialize_embedding_model()

#     # Function to get the best available device
#     def get_device(self):
#         if torch.cuda.is_available():
#             try:
#                 torch.cuda.empty_cache()  # Clear unused memory
#                 return torch.device("cuda")
#             except RuntimeError:
#                 print("GPU out of memory. Falling back to CPU.")
#         return torch.device("cpu")

#     # Try initializing the model on GPU first, fallback to CPU if OOM error occurs
#     def initialize_model(self):
#         try:
#             pipe = pipeline(
#                 "text-generation",
#                 model="Qwen/Qwen2-7B-Instruct",
#                 device=0 if self.device.type == "cuda" else -1
#             )
#             print(f"Using device: {self.device}")
#             return pipe
#         except RuntimeError as e:
#             if "out of memory" in str(e).lower():
#                 print("GPU ran out of memory. Clearing memory and retrying...")
#                 torch.cuda.empty_cache()
#                 import gc
#                 gc.collect()

#                 # Try again on GPU
#                 try:
#                     pipe = pipeline(
#                         "text-generation",
#                         model="Qwen/Qwen2-7B-Instruct",
#                         device=0
#                     )
#                     print("Retrying on GPU succeeded.")
#                     return pipe
#                 except RuntimeError as e:
#                     print("GPU still has OOM error. Switching to CPU.")
#                     self.device = torch.device("cpu")
#                     return pipeline("text-generation", model="Qwen/Qwen2-7B-Instruct", device=-1)
#             else:
#                 raise e


#     # Method to get the API key from the .env file
#     def get_api_key(self, api_name):
#         env_path = "../.dummy_env"  # Adjust the path as needed
#         load_dotenv(env_path)  # Load the environment variables
#         return os.getenv(api_name)

#     # Initialize Pinecone and return the index
#     def initialize_pinecone(self):
#         # Access the Pinecone API key
#         pinecone_api_key = self.get_api_key("PINECONE_API_KEY")
#         pinecone = Pinecone(api_key=pinecone_api_key)
#         index_name = "recursive-text-chunks-new"
#         index = pinecone.Index(index_name)
#         return index

#     # Initialize the embedding model
#     def initialize_embedding_model(self):
#         return SentenceTransformer("all-MiniLM-L6-v2")

#     # Query Pinecone
#     def query_pinecone(self, query_text, top_k=5):
#         # Check if the embedding model and index are initialized
#         if not hasattr(self, 'embedding_model') or not hasattr(self, 'index'):
#             raise ValueError("Embedding model and index must be defined before running the function.")

#         # Query Pinecone to get relevant chunks
#         query_embedding = self.embedding_model.encode(query_text).tolist()
#         return self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

#     # Generate answer
#     def generate_answer(self, query_text):
#         # Query Pinecone for relevant context
#         results = self.query_pinecone(query_text)

#         # Concatenate the top results into a context for the Llama model
#         context = "\n".join([match['metadata']['text'] for match in results.get("matches", [])])

#         # Read the reasoning prompt from file
#         with open("prompt.txt", "r") as f:
#             reasoning_prompt = f.read().strip()

#         # Construct the prompt
#         prompt = (
#             f"{reasoning_prompt}\n\n"
#             "Context:\n"
#             f"{context}\n\n"
#             f"Question: {query_text}\n"
#         )

#         # Use the pipeline to generate the answer
#         generated = self.pipe(
#             prompt,
#             max_new_tokens=200,
#             num_return_sequences=1,
#             temperature=0.7,
#             top_p=0.9
#         )

#         # Extract the generated answer
#         full_generated_text = generated[0]['generated_text']
#         final_answer = full_generated_text[len(prompt):].strip()
        
#         return final_answer
