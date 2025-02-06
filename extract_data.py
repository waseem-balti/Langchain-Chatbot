from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Load webpage content
url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
docs = loader.load()

# Split text into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# Extract raw text from chunks
documents = [doc.page_content for doc in chunks]

# Save extracted data (optional, for debugging)
with open("extracted_data.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n\n")

print("✅ Step 1: Data extraction completed!")
