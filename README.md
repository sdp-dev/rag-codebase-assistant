# RAG Codebase Assistant

Ask questions about any Python GitHub repository in plain English. The assistant clones a repo, parses it into functions, classes, and methods with tree-sitter, indexes them in a FAISS vector store, and answers questions with a local LLM (via Ollama), citing the file and line numbers it pulled from.

Everything runs locally: no API keys, and your code never leaves your machine.

## Features

- **Index any public GitHub repo** by pasting its URL. Re-indexing pulls the latest changes.
- **Code-aware chunking** using tree-sitter: each function, class, and method becomes its own chunk, with its docstring, file path, and line range.
- **Semantic search** with `all-mpnet-base-v2` sentence embeddings and a cosine-similarity FAISS index.
- **Multi-repo search**: index several repositories and choose which ones each question searches.
- **Grounded answers**: the LLM is prompted to answer from the retrieved code and cite `file:line` locations. Every answer shows the snippets it was based on.
- **Conversational**: follow-up questions keep the chat history.
- **Persistent index** saved to disk, so repos stay indexed between sessions.

## How it works

```
GitHub URL
   │  GitPython clone / pull
   ▼
Python files ── tree-sitter ──▶ chunks (function / class / method + docstring + location)
                                   │  sentence-transformers (all-mpnet-base-v2)
                                   ▼
                            FAISS index (inner product on normalized vectors)
                                   │
Question ──▶ embed ──▶ top-k similar chunks (filtered by selected repos)
                                   │  formatted as context
                                   ▼
                       LangChain + Ollama (llama3) ──▶ answer with citations
```

## Project structure

```
app.py                      Streamlit UI: add/re-index/delete repos, chat
config.py                   Paths, embedding model, top-k, LLM model
src/
  ingestion/
    github_client.py        Clone or pull repositories
    repo_manager.py         Track indexed repos in data/metadata.json
  parsing/
    python_parser.py        tree-sitter parser → CodeChunk objects
    code_chunker.py         Walk a repo, skip venvs/build dirs/tests
  indexing/
    embedder.py             sentence-transformers wrapper
    vector_store.py         FAISS index + chunk metadata, save/load, per-repo removal
  retrieval/
    retriever.py            Top-k search and context formatting
  conversation/
    chain.py                LangChain chat over Ollama with history
    prompts.py              System prompt and context template
tests/                      Parser, embedder, vector store, retriever, end-to-end
```

## Getting started

### Option 1: Docker (recommended)

Runs the app and Ollama together.

```bash
docker compose up -d --build
./setup-model.sh        # pulls llama3 into the Ollama container (one time)
```

Open http://localhost:8501.

### Option 2: Run locally

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt langchain-ollama

ollama pull llama3      # in another terminal, make sure `ollama serve` is running
streamlit run app.py
```

## Usage

1. Paste a GitHub URL (e.g. `https://github.com/owner/repo`) in the sidebar and click **Index Repository**.
2. Ask a question in the chat, for example:
   - "Where is authentication handled?"
   - "How does the `VectorStore` remove a repository?"
   - "What does `parse_source` return?"
3. Expand **Retrieved Code Snippets** under an answer to see exactly which code it used.

Use **Select repos to search** to limit questions to certain repositories, and the **Results to retrieve** slider to control how many snippets go into each prompt.

## Configuration

Edit `config.py`:

| Setting | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | sentence-transformers model for embeddings |
| `DEFAULT_TOP_K` | `5` | Snippets retrieved per question |
| `LLM_MODEL` | `llama3` | Any Ollama model, e.g. `mistral` or `codellama` |

Set `OLLAMA_HOST` to point at a different Ollama server (defaults to `http://localhost:11434`).

## Tests

```bash
pip install pytest
python -m pytest
```

## Limitations

- Only Python files are parsed; test files are skipped by default.
- Private repositories need git credentials configured on the machine running the app.
- Removing a repository rebuilds the index from the remaining chunks, which re-embeds them.

## Tech stack

Python · Streamlit · LangChain · Ollama · sentence-transformers · FAISS · tree-sitter · GitPython · Docker
