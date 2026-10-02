import os
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

DATA_DIR = Path("1000+ PDF_Invoice_Folder")
OUTPUT_FILE = "invoices_metadata.json"
MAX_FILES = 20


# 1. Schéma des métadonnées à extraire
class InvoiceMetadata(BaseModel):
    invoice_number: Optional[str] = Field(None, description="Invoice number / ID")
    date: Optional[str] = Field(None, description="Invoice date, format YYYY-MM-DD if possible")
    bill_to: Optional[str] = Field(None, description="Name/address of the 'Bill To' party")
    ship_to: Optional[str] = Field(None, description="Name/address of the 'Ship To' party")
    shipping: Optional[float] = Field(None, description="Shipping cost as a number")
    discount: Optional[float] = Field(None, description="Discount amount as a number (0 if none)")
    total: Optional[float] = Field(None, description="Final total amount as a number")


# 2. LLM avec sortie structurée
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0,
)
structured_llm = llm.with_structured_output(InvoiceMetadata)

prompt = ChatPromptTemplate.from_template(
    """
    You are an expert at extracting data from invoices.
    Extract the requested fields from the invoice text below.
    If a field is not present, return null. Do not invent values.
    Numbers must be plain numbers (no currency symbols, no commas).

    Invoice text:
    {text}
    """
)

chain = prompt | structured_llm


# 3. Lecture d'un PDF
def read_pdf(path: Path) -> str:
    pages = PyPDFLoader(str(path)).load()
    return "\n".join(p.page_content for p in pages)


# 4. Traitement des 20 premiers fichiers
def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))[:MAX_FILES]
    print(f"{len(pdf_files)} fichiers à traiter")

    results = []
    for i, pdf in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] {pdf.name}")
        try:
            text = read_pdf(pdf)
            if not text.strip():
                raise ValueError("PDF sans texte (scanné ?)")
            metadata = chain.invoke({"text": text})
            results.append({"file": pdf.name, **metadata.model_dump()})
        except Exception as e:
            print(f"   Erreur : {e}")
            results.append({"file": pdf.name, "error": str(e)})

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés dans {OUTPUT_FILE}")


if __name__ == "__main__":
    main()