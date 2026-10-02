# 🧾 invoice-rag-assistant

A conversational assistant based on **RAG** (Retrieval-Augmented Generation) to query **PDF invoices**: automatic metadata extraction, indexing in Qdrant, then natural-language chat.

## Features

- **Metadata extraction** from PDF invoices using an LLM (structured output): invoice number, date, total, shipping, discount, `bill_to`, `ship_to`.
- **Ingestion**: PDFs are split into chunks, embedded, and stored in **Qdrant** with the extracted metadata attached to every chunk.
- **RAG chat**: question rewriting, semantic retrieval, and answers generated strictly from the retrieved context.
- **Metadata filtering** (e.g. restrict the search to a specific invoice).
- **Web interface** built with Streamlit, in addition to the terminal chat.

## Architecture

```
PDF invoices (data/)
        │
        ▼
metadata_extraction.py   →  invoice_number, date, total, shipping, discount, bill_to, ship_to
        │
        ▼
Ingestion.py             →  chunks + embeddings + metadata  →  Qdrant
        │
        ▼
chat.py / app.py         →  question → rewrite → retrieval (with filter) → LLM → answer
```

## Project Structure

```
invoice-rag-assistant/
├── data/                    # PDF invoices (only the first 20 files are processed)
├── metadata_extraction.py   # LLM-based metadata extraction
├── Ingestion.py             # ingestion into Qdrant
├── chat.py                  # terminal RAG chat
├── app.py                   # Streamlit web interface
├── requirements.txt
├── .env.example
└── README.md
```

## Tech Stack

- Python 3.10+
- [LangChain](https://www.langchain.com/) (`langchain-groq`, `langchain-huggingface`, `langchain-qdrant`)
- [Groq](https://groq.com/) (LLM `openai/gpt-oss-20b`)
- [Qdrant](https://qdrant.tech/) (vector database)
- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings
- Streamlit (web interface)

## Prerequisites

- Python and pip
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) to run Qdrant
- A Groq API key

## Installation

```bash
git clone https://github.com/cherni2003/invoice-rag-assistant.git
cd invoice-rag-assistant

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
```

## Configuration

env add your key:

```
GROQ_API_KEY=your_key_here
```

> ⚠️ Never commit the `.env` file.

## Usage

### 1. Start Qdrant

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

Check it is running: <http://localhost:6333/dashboard>

### 2. Extract metadata (optional, standalone test)

```bash
python metadata_extraction.py
```

Generates `invoices_metadata.json` for the first 20 invoices in the `data/` folder.

### 3. Ingest invoices into Qdrant

```bash
python Ingestion.py
```

Creates the `invoices_rag` collection.

### 4. Chat with your invoices

In the terminal:

```bash
python chat.py
```

Commands: `/invoice <number>` to filter on one invoice, `/all` to remove the filter, `/quit` to exit.

Or through the web interface:

```bash
streamlit run app.py
```

Then open <http://localhost:8501>.

## Example Questions

- *What is the total of invoice 12345?*
- *Who is the invoice billed to?*
- *What was the shipping cost and the discount?*
- *On which date was this invoice issued?*

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `WinError 10061` (connection refused) | Qdrant is not running | Start Docker Desktop and the Qdrant container |
| `404 Collection invoices_rag not found` | Ingestion has not been run | Run `python Ingestion.py` |
| "I don't know" answers | Wrong filter or badly formatted invoice number | Use `/all` or check the payload in the Qdrant dashboard |

## Limitations

- Only the **first 20 PDFs** are processed (`MAX_FILES` parameter).
- Scanned PDFs (images) require an OCR step, which is not included.
- Extraction quality depends on the LLM used.

## Author

Oumaima Cherni, TEK-UP, Data Science & AI
