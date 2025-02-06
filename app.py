from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


app = Flask(__name__)
api = Api(app)

# Load FAISS vector store with safe deserialization
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

class QueryAPI(Resource):
    def post(self):
        try:
            # Get user query
            data = request.get_json()
            query = data.get("query", "")

            if not query:
                return jsonify({"error": "Query parameter is required!"}), 400

            # Search FAISS vector store
            results = vectorstore.similarity_search(query, k=3)  # Top 3 matches

            # Format response
            response = [{"text": result.page_content, "metadata": result.metadata} for result in results]
            return jsonify({"query": query, "results": response})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Define API route
api.add_resource(QueryAPI, "/query")

if __name__ == "__main__":
    app.run(debug=True)