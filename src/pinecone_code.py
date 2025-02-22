from pinecone import Pinecone, ServerlessSpec
import os
import time
import json
import dotenv
import google.generativeai as genai
import nltk
from nltk.tokenize import sent_tokenize
import uuid
nltk.download('punkt')


class Pinecone_code:
    def __init__(self)-> None:
        dotenv.load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone = Pinecone(api_key=self.pinecone_api_key) 

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=self.gemini_api_key)       

        with open('config.json') as f:
            data = json.load(f)
        
        self.pinecone_index_name = data['pinecone_index_name'] # Jenitza
        self.pinecone_namespace = data['pinecone_namespace'] # default 
        self.pinecone_vector_dimension = data['pinecone_vector_dimension'] # 768 and should match the gemini encoding dimension
        self.pinecone_upsert_batch_limit = data['pinecone_upsert_batch_limit'] # 50

        self.gemini_embedding_model = data['gemini_embedding_model'] # "models/embedding-001"

    def _check_index_already_exists(self) -> bool:
        return self.pinecone_index_name in self.pinecone.list_indexes() 

    def _get_index(self):
        return self.pinecone.Index(self.pinecone_index_name)

    def _create_new_index(self):
        self.pinecone.create_index(
                    name=self.pinecone_index_name,
                    metric='cosine',
                    dimension=self.pinecone_vector_dimension,
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1',
                        # pod_type="p1.x1", # Future use
                    ),
                )

    def _create_index(self, force_delete=False)-> None:
        if self._check_index_already_exists():
            if force_delete:
                self.pinecone.delete_index(name=self.index_name)
                time.sleep(30) # Wait for the index to be deleted
                self._create_new_index()
            else:
                print(f"Index {self.index_name} already exists")
        else:
            self._create_new_index()


    def _embed_text(self, text:str)-> list:
        '''
        Embed the text using the Gemini API and return the embedding vector
        '''
        #Ref: https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/tutorials/document_search.ipynb
        retry = 0
        while retry < 3:
            try:
                response = genai.embed_content(content=text, model=self.gemini_embedding_model,
                                                task_type="retrieval_document")
                return response['embedding']
            
            except Exception as e:
                print("Error embedding text: ", e)
                retry += 1
                time.sleep(2** retry)



    def _document_chunking(self, document: str, chunk_size: int) -> list:
        '''
        Chunk the document into smaller pieces, breaks into senteces
        '''
        sentences = [sentence.strip() for sentence in sent_tokenize(str(document)) if sentence.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            space_needed = 1 if current_chunk else 0
            total_length = current_length + space_needed + sentence_length

            if total_length <= chunk_size:
                current_chunk.append(sentence)
                current_length += space_needed + sentence_length
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    chunks.append(sentence)
                    current_chunk = []
                    current_length = 0

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _batch_upsert(self, vectors: list, index: Pinecone.Index) -> None:
        retry = 0
        while retry < 3:
            try:
                index.upsert(
                    vectors=vectors,
                    namespace=self.pinecone_namespace,
                )
                return
            except Exception as e:
                print("Error upserting into Pinecone: ", e)
                retry += 1
                time.sleep(2**retry)

    def pinecone_upsert(self, document: str) -> None:
        '''
        Takes a document and upserts it into the Pinecone database
        '''
        # Chunk the document
        chunks = self._document_chunking(document=document, chunk_size=150)
        index = self._get_index() 

        vectors = []
        for chunk in chunks:
            vector_id = str(uuid.uuid4())
            vector_values = self._embed_text(chunk) # Generate the embedding vector : Convert text --> vector embedding
            # Prepare metadata
            metadata = {'text': chunk}
            # Create the vector dictionary
            vector = {
                'id': vector_id,
                'values': vector_values,
                'metadata': metadata
            }
            vectors.append(vector) # temporary store the vectors
            
            # Batch upsert when reaching the limit
            if len(vectors) >= self.pinecone_upsert_batch_limit:  # once the limit is reached,  tell pinecone to upsert
                time.sleep(2)
                self._batch_upsert(vectors, index)
                vectors = []

        # Upsert any remaining vectors
        if vectors:
            self._batch_upsert(vectors, index)


    def query_pinecone(self, query: str, top_k: int = 5) -> list:
        '''
        Get the top k results from Pinecone for the given query
        '''
        index = self._get_index()
        query_vector = self._embed_text(query)
        
        if not query_vector:
            print("Failed to generate query embedding")
            return []
        
        try:
            return index.query(
                        vector=query_vector,  
                        namespace=self.pinecone_namespace,
                        include_metadata=True,
                        top_k=top_k
                    )
        except Exception as e:
            print("Error querying Pinecone: ", e)
            return []
        
             