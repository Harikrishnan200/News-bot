# NewsBot

NewsBot is a small, local question-answering API for news articles. You provide one or more article URLs, and the application:

1. Downloads and extracts the article text.
2. Splits the text into smaller overlapping sections.
3. Creates vector embeddings for those sections.
4. Stores the sections and embeddings in a local ChromaDB database.
5. Finds the most relevant sections when you ask a question.
6. Uses a local Ollama language model to write an answer from those sections.

This is a Retrieval-Augmented Generation (RAG) application. It does not search the internet when you ask a question; it answers from the articles that have already been processed.

## What is used in this project?

| Technology | Purpose |
| --- | --- |
| Python | Application programming language |
| FastAPI | Creates the HTTP API and its endpoints |
| Uvicorn | Runs the FastAPI application |
| Pydantic | Validates request and response data |
| `python-dotenv` | Loads configuration from a `.env` file |
| LangChain | Connects the loader, text splitter, vector store, prompt, and model |
| `UnstructuredURLLoader` | Downloads and extracts text from URLs |
| `RecursiveCharacterTextSplitter` | Breaks long article text into smaller chunks |
| Ollama | Runs the language model and embedding model locally |
| `phi3` | Local chat model used to generate answers |
| `nomic-embed-text` | Local embedding model used for similarity search |
| ChromaDB | Stores article chunks and their vector embeddings on disk |

## Project structure

```text
News-bot/
├── README.md
└── news-bot/
    └── main.py       # FastAPI application and RAG workflow
```

The `chroma_db/` directory is created automatically after the first successful URL-processing request. It contains the local vector database and is not present in a fresh checkout.

## How the application works

### 1. Add article URLs

The `POST /process-urls` endpoint receives a list of URLs. `UnstructuredURLLoader` loads their content, and the text splitter divides it into chunks of up to 1,000 characters with 200 characters of overlap.

The overlap helps preserve context when an important sentence falls near the boundary between two chunks.

### 2. Create and store embeddings

Each chunk is converted into a numerical representation called an embedding by Ollama's `nomic-embed-text` model. ChromaDB stores both the original text and its embedding in the local `./chroma_db` directory.

### 3. Retrieve relevant text

When a question is sent to `POST /ask`, ChromaDB performs a similarity search and returns the four most relevant chunks (`k=4`).

### 4. Generate the answer

The retrieved chunks are placed into a prompt and sent to the local `phi3` model. The prompt tells the model to use only the supplied context and list the sources used.

Because the model and embeddings run through Ollama on the local machine, the application does not need a cloud LLM API key.

## Prerequisites

- Python 3.10 or newer is recommended.
- Ollama must be installed and running.
- The required Ollama models must be available locally:

```bash
ollama pull phi3
ollama pull nomic-embed-text
```

Ollama normally listens on `http://localhost:11434`, which is the URL configured in `main.py`.

## Installation

From the repository root, create and activate a virtual environment:

### macOS/Linux

```bash
cd news-bot
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
cd news-bot
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the Python packages used by the application:

```bash
pip install fastapi uvicorn pydantic python-dotenv \
  langchain-ollama langchain-community langchain-text-splitters \
  langchain-chroma langchain-core chromadb unstructured
```

There is currently no `requirements.txt` file. If you want to capture the installed versions for repeatable setup, run:

```bash
pip freeze > requirements.txt
```

## Start the API

Make sure Ollama is running, then start the application from the `news-bot/` directory:

```bash
python main.py
```

The API will be available at:

- API root: <http://localhost:8000>
- Interactive Swagger documentation: <http://localhost:8000/docs>
- Alternative ReDoc documentation: <http://localhost:8000/redoc>

The server uses auto-reload, so it restarts when `main.py` changes.

You can also start it directly with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API endpoints

### `GET /`

Checks that the API is running and reports the configured local models.

Example:

```bash
curl http://localhost:8000/
```

Example response:

```json
{
  "message": "NewsBot running on local Ollama",
  "llm": "phi3",
  "embeddings": "nomic-embed-text"
}
```

### `POST /process-urls`

Loads and stores one or more article URLs.

Request:

```bash
curl -X POST http://localhost:8000/process-urls \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/article-1",
      "https://example.com/article-2"
    ]
  }'
```

The response reports how many URLs were accepted and how many text chunks were stored. Run this endpoint before asking questions.

### `POST /ask`

Asks a question about the processed articles.

Request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main points in the articles?"
  }'
```

Example response:

```json
{
  "answer": "The articles mainly discuss ...",
  "sources": [
    "https://example.com/article-1"
  ]
}
```

If no articles have been processed, the endpoint returns an error asking you to call `/process-urls` first.

### `DELETE /clear`

Deletes the local ChromaDB data so you can start with an empty article collection.

```bash
curl -X DELETE http://localhost:8000/clear
```

This removes the generated `chroma_db/` directory. Articles will need to be processed again afterward.

## Using the Swagger UI

For a beginner-friendly way to test the API:

1. Start the server with `python main.py`.
2. Open <http://localhost:8000/docs> in a browser.
3. Expand `POST /process-urls`, click **Try it out**, and submit article URLs.
4. Expand `POST /ask`, enter a question, and submit it.
5. Inspect the generated answer and returned source URLs.

## Configuration

The main settings are currently defined near the top of `news-bot/main.py`:

```python
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "news_articles"

llm = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434",
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)
```

The application uses a shared ChromaDB client for its lifetime. This avoids repeatedly opening a database connection for each request.

There are no required environment variables at the moment. `load_dotenv()` is included so environment-based configuration can be added later.

## Important limitations

- There is no frontend; the project currently exposes an API only.
- There is no authentication or authorization. Do not expose this API publicly without adding security controls.
- Article extraction depends on the target website and may fail for pages that require JavaScript, block automated requests, or do not contain readable article text.
- The answer quality depends on the downloaded article content, the retrieved four chunks, and the local `phi3` model.
- Processing the same URL repeatedly can add duplicate chunks to the collection.
- The local ChromaDB directory is runtime data and should generally not be committed to source control.
- Error details from failed requests are returned in the API response, which is useful during development but may reveal internal details in production.

## Troubleshooting

### Ollama connection errors

Check that Ollama is running and that both models are installed:

```bash
ollama list
ollama serve
```

If Ollama uses a different host or port, update the `base_url` values in `main.py`.

### No relevant information found

Confirm that:

1. `/process-urls` completed successfully.
2. The URL contains readable article text.
3. The question relates to the stored articles.

To remove old data and rebuild the collection, call `DELETE /clear`, then process the URLs again.

### Import errors

Make sure the virtual environment is active and install the packages from the installation section again. The URL loader may also require additional system dependencies for certain file or page formats.

## Possible next improvements

- Add a `requirements.txt` or `pyproject.toml` with pinned versions.
- Add URL deduplication and article metadata such as title and publication date.
- Add request authentication and rate limiting.
- Add structured logging and more specific error handling.
- Add tests for URL processing, retrieval, and API responses.
- Add a frontend for submitting URLs and displaying answers.
- Move model names, Ollama URL, chunk settings, and database paths into environment variables.

