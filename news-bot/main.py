import os
import shutil
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
import chromadb

load_dotenv()

app = FastAPI(title="NewsBot API", version="2.0.0")

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "news_articles"

llm = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434",
    temperature=0.9,
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

PROMPT = ChatPromptTemplate.from_template("""
You are a helpful news assistant. Use ONLY the context below to answer the question.
At the end, list the sources you used.

Context:
{context}

Question: {question}

Answer:
""")

# Shared ChromaDB client — one connection for the whole app lifetime
chroma_client: Optional[chromadb.PersistentClient] = None

def get_chroma_client() -> chromadb.PersistentClient:
    """Return the shared ChromaDB client, creating it if needed."""
    global chroma_client
    if chroma_client is None:
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    return chroma_client


class URLRequest(BaseModel):
    urls: List[str]

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]


def process_and_store_urls(urls: List[str]) -> int:
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    if not documents:
        raise ValueError("No content could be loaded from the provided URLs.")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=get_chroma_client(),  # use shared client
    )

    return len(docs)


@app.get("/")
def home():
    return {
        "message": "NewsBot running on local Ollama",
        "llm": "phi3",
        "embeddings": "nomic-embed-text"
    }


@app.post("/process-urls")
def process_urls(request: URLRequest):
    valid_urls = [url for url in request.urls if url.strip()]
    if not valid_urls:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one valid URL."
        )
    try:
        num_chunks = process_and_store_urls(valid_urls)
        return {
            "status": "success",
            "message": f"Processed {len(valid_urls)} URL(s), stored {num_chunks} chunks.",
            "urls_processed": valid_urls
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URLs: {str(e)}"
        )


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not os.path.exists(CHROMA_DB_DIR):
        raise HTTPException(
            status_code=400,
            detail="No articles processed yet. Call /process-urls first."
        )
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            client=get_chroma_client(),  # use shared client
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(request.question)

        if not relevant_docs:
            return AnswerResponse(
                answer="No relevant information found in the processed articles.",
                sources=[]
            )

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources_list = list({
            doc.metadata.get("source", "Unknown")
            for doc in relevant_docs
        })

        chain = PROMPT | llm
        response = chain.invoke({
            "context": context,
            "question": request.question
        })

        answer = response.content if hasattr(response, "content") else str(response)
        return AnswerResponse(answer=answer, sources=sources_list)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )


@app.delete("/clear")
def clear_database():
    global chroma_client
    if not os.path.exists(CHROMA_DB_DIR):
        return {"status": "info", "message": "Database was already empty."}

    try:
        # Reset via the shared client (deletes all collections cleanly)
        if chroma_client is not None:
            chroma_client.reset()
            chroma_client = None  # force re-creation on next request

        # Now safe to delete the folder on Windows
        shutil.rmtree(CHROMA_DB_DIR)
        return {"status": "success", "message": "ChromaDB cleared."}

    except Exception as e:
        return {
            "status": "error",
            "message": f"Could not clear database: {str(e)}. Try restarting the server."
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)