import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# ✅ Load extracted data
EXTRACTED_DATA_FILE = "extracted_data.txt"

if not os.path.exists(EXTRACTED_DATA_FILE):
    raise FileNotFoundError(f"❌ Error: {EXTRACTED_DATA_FILE} not found! Run extract_data.py first.")

with open(EXTRACTED_DATA_FILE, "r", encoding="utf-8") as f:
    documents = f.readlines()

# ✅ Remove empty lines and strip whitespace
documents = [doc.strip() for doc in documents if doc.strip()]

# ✅ Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Generate embeddings
embeddings = embedding_model.encode(documents, convert_to_numpy=True)

# ✅ Store embeddings in FAISS
vectorstore = FAISS.from_texts(
    texts=documents,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)


# ✅ Save FAISS index
vectorstore.save_local("faiss_index")

print("✅ Step 2: Embeddings created and stored in FAISS successfully!")
