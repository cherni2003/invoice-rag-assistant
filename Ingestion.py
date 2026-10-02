import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

from metadata_extraction import chain, read_pdf, MAX_FILES, DATA_DIR  # réutilise ton fichier

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices_rag"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

all_chunks = []
for pdf in sorted(DATA_DIR.glob("*.pdf"))[:MAX_FILES]:
    # 1. métadonnées (via metadata_extraction)
    meta = chain.invoke({"text": read_pdf(pdf)}).model_dump()
    meta["source"] = str(pdf)

    # 2. chunks, chacun hérite des métadonnées
    docs = PyPDFLoader(str(pdf)).load()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata.update(meta)
    all_chunks.extend(chunks)

# 3. stockage dans Qdrant
QdrantVectorStore.from_documents(
    all_chunks,
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)
print(f"{len(all_chunks)} chunks ingérés")