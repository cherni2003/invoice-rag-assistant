import os
os.environ["HF_HUB_OFFLINE"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "invoices_rag"   # même nom que dans Ingestion.py

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant that answers questions about invoices.
    Always answer using only the context below.
    If the answer is not in the context, say "I don't know".

    Question:
    {question}

    Context:
    {context}
    """
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """
    Rewrite the user's question into a concise search query.
    Do not answer the question. Keep the important keywords.

    Question:
    {question}

    Search query:
    """
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    url=QDRANT_URL,
    collection_name=COLLECTION_NAME,
)


def build_retriever(invoice_number=None):
    """Retriever, avec filtre optionnel sur le numéro de facture."""
    search_kwargs = {"k": 5}
    if invoice_number:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.invoice_number",
                    match=MatchValue(value=invoice_number),
                )
            ]
        )
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def format_docs(documents):
    # On inclut les métadonnées pour que le LLM voie total, date, etc.
    parts = []
    for doc in documents:
        m = doc.metadata
        header = (
            f"[Invoice {m.get('invoice_number')} | date: {m.get('date')} | "
            f"total: {m.get('total')} | shipping: {m.get('shipping')} | "
            f"discount: {m.get('discount')} | bill to: {m.get('bill_to')} | "
            f"ship to: {m.get('ship_to')} | file: {m.get('source')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def build_rag_chain(retriever):
    return (
        {
            "context": rewrite_chain | retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    invoice_filter = None
    rag_chain = build_rag_chain(build_retriever())

    print("Chat factures. Commandes : /invoice <numéro> (filtrer), /all (tout), /quit")
    while True:
        question = input("Ask something : ").strip()
        if not question:
            continue

        if question.lower() in ["/quit", "/bye"]:
            print("Exiting the program.")
            break

        if question.lower().startswith("/invoice "):
            invoice_filter = question.split(maxsplit=1)[1].strip()
            rag_chain = build_rag_chain(build_retriever(invoice_filter))
            print(f"Filtre actif : facture {invoice_filter}")
            continue

        if question.lower() == "/all":
            invoice_filter = None
            rag_chain = build_rag_chain(build_retriever())
            print("Filtre supprimé.")
            continue

        answer = rag_chain.invoke(question)
        print("AI :", answer, "\n")


if __name__ == "__main__":
    main()