import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    stream: bool = False

class ModelRequest(BaseModel):
    model_name: str

class RAGSystem:
    def __init__(self, embeddings_dir: str, api_key: str,
                 model_name: str = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                 llm_model: str = "deepseek/deepseek-r1"):
        self.api_key = api_key
        self.llm_model = llm_model
        self.chunks, self.embeddings = self.load_embeddings(embeddings_dir)
        self.embedding_model = SentenceTransformer(model_name)
        self.embeddings_index = np.array(self.embeddings)
        logger.info(f"RAG system initialized with {len(self.chunks)} chunks")

    def load_embeddings(self, embeddings_dir: str):
        all_chunks = []
        all_embeddings = []
    
        for file in os.listdir(embeddings_dir):
            if file.startswith('embeddings_') and file.endswith('.json'):
                try:
                    file_path = os.path.join(embeddings_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks_list = json.load(f)
                        if not isinstance(chunks_list, list):
                            continue
                    
                        for chunk in chunks_list:
                            metadata = {
                                "source": chunk.get("source", "unknown"),
                                "page": chunk.get("page", 0),
                                "fragment_id": chunk.get("fragment_id", "")
                            }
                            all_chunks.append({
                                "text": chunk["text"],
                                "metadata": metadata
                            })
                            all_embeddings.append(chunk["embedding"])
                except Exception as e:
                    logger.error(f"Error loading {file}: {str(e)}")
        return all_chunks, all_embeddings

    def find_relevant_chunks(self, query: str, top_k: int = 3):
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        norms = np.linalg.norm(self.embeddings_index, axis=1) * np.linalg.norm(query_embedding)
        norms[norms == 0] = 1e-10  # Avoid division by zero
        similarities = np.dot(self.embeddings_index, query_embedding) / norms
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [{
            "text": self.chunks[idx]["text"],
            "metadata": self.chunks[idx]["metadata"],
            "score": float(similarities[idx])
        } for idx in top_indices]

    def generate_response(self, query: str, context_chunks: List[Dict], stream: bool = False) -> str:
        context = "\n\n".join(
            [f"[Document: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}]\n{chunk['text']}" 
             for chunk in context_chunks]
        )
        
        messages = [
            {
                "role": "system",
                "content": f"You are R.A.G, an efficient corporate knowledge assistant. "
                           f"Answer ONLY based on the provided context!\n\nCONTEXT:\n{context}"
            },
            {"role": "user", "content": query}
        ]
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": stream
        }
        
        try:
            if stream:
                return self.stream_response(data, headers)
            return self.non_stream_response(data, headers)
        except Exception as e:
            return f"Generation error: {str(e)}"

    def stream_response(self, data: dict, headers: dict) -> str:
        full_response = []
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            stream=True
        ) as response:
            if response.status_code != 200:
                return f"API error: {response.status_code}"
                
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                    
                chunk_str = chunk.decode('utf-8').replace('data: ', '').strip()
                if chunk_str == "[DONE]":
                    break
                    
                try:
                    chunk_json = json.loads(chunk_str)
                    if "choices" in chunk_json:
                        content = chunk_json["choices"][0]["delta"].get("content", "")
                        full_response.append(content)
                except json.JSONDecodeError:
                    continue
        return ''.join(full_response)

    def non_stream_response(self, data: dict, headers: dict) -> str:
        data["stream"] = False
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        if response.status_code != 200:
            return f"API error: {response.status_code}"
        return response.json()["choices"][0]["message"]["content"]

    def ask(self, question: str, top_k: int = 3, stream: bool = False) -> Dict[str, Any]:
        relevant_chunks = self.find_relevant_chunks(question, top_k=top_k)
        answer = self.generate_response(question, relevant_chunks, stream=stream)
        return {
            "question": question,
            "answer": answer,
            "contexts": relevant_chunks
        }
    
    def get_source_content(self, filename: str) -> str:
        """Получить содержимое источника по имени файла"""
        content = []
        for chunk in self.chunks:
            if chunk['metadata']['source'] == filename:
                content.append(chunk['text'])
        return "\n\n".join(content)

    def get_unique_sources(self) -> List[str]:
        sources = set()
        for chunk in self.chunks:
            source = chunk["metadata"]["source"]
            if source and source != "unknown":
                sources.add(source)
        return list(sources)

# Global RAG system
rag_system: Optional[RAGSystem] = None

@app.on_event("startup")
def startup_event():
    global rag_system
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY environment variable not set")
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    
    # Создаем директорию для эмбеддингов, если её нет
    embeddings_dir = "embedding"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    rag_system = RAGSystem(
        embeddings_dir=embeddings_dir,
        api_key=api_key,
        llm_model="deepseek/deepseek-r1"
    )
    logger.info("RAG system initialized successfully")

@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        result = rag_system.ask(
            question=request.question,
            top_k=request.top_k,
            stream=request.stream
        )
        return result
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_model")
async def set_model(request: ModelRequest):
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    rag_system.llm_model = request.model_name
    return {"message": f"Model changed to {request.model_name}"}

@app.get("/sources")
async def get_sources():
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return {"sources": rag_system.get_unique_sources()}

@app.get("/source_content")
async def get_source_content(filename: str):
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        content = rag_system.get_source_content(filename)
        return {"content": content}
    except Exception as e:
        logger.error(f"Error getting source content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/system_info")
async def system_info():
    if not rag_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return {
        "model": rag_system.llm_model,
        "chunks_loaded": len(rag_system.chunks)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)