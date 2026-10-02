import streamlit as st

from chat import build_retriever, build_rag_chain  # réutilise la logique de chat.py

st.set_page_config(page_title="Invoices RAG Chat", page_icon="🧾")
st.title("🧾 Chat sur les factures")


@st.cache_resource
def get_chain(invoice_number):
    """Crée la chaîne RAG (mise en cache selon le filtre choisi)."""
    return build_rag_chain(build_retriever(invoice_number or None))


# --- Barre latérale : filtre optionnel ---
with st.sidebar:
    st.header("Options")
    invoice_filter = st.text_input("Filtrer par numéro de facture", placeholder="ex. 12345")
    if st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Historique ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Nouvelle question ---
question = st.chat_input("Pose une question sur tes factures...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche en cours..."):
            try:
                chain = get_chain(invoice_filter.strip() or None)
                answer = chain.invoke(question)
            except Exception as e:
                answer = f"Erreur : {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})