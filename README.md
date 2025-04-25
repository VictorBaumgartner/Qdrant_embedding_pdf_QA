
# 📚 PDF QA System with Langchain, Qdrant & BAAI Embeddings

This project is a **PDF Question-Answering System** that allows you to perform semantic search and QA on a PDF file using:

- 🔍 [Langchain](https://github.com/langchain-ai/langchain)
- 🧠 [BAAI bge-large-en Embeddings](https://huggingface.co/BAAI/bge-large-en)
- 🗃️ [Qdrant Vector Database](https://qdrant.tech/)
- 🐳 Docker for local Qdrant deployment

---

## ✨ Features

- Load and chunk PDF files using `PyPDFLoader`
- Embed text using `BAAI/bge-large-en` model from Hugging Face
- Store and search embeddings in Qdrant (Cosine similarity)
- Ask questions and retrieve the most relevant sections (chunks) of the PDF

---

## 🧰 Requirements

Make sure you have the following installed:

- Docker
- Python 3.8+
- `pip` or `conda`

Install Python dependencies:

```bash
pip install langchain qdrant-client huggingface_hub pypdf
```

---

## 🚀 Setup

### 1. Start Qdrant with Docker

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Add your PDF

Place your file as `data.pdf` in the root folder (or change the path in the code).

---

## 🧠 Embedding & Indexing

The script will:

1. Load `data.pdf`
2. Split it into overlapping chunks
3. Generate embeddings using the `BAAI/bge-large-en` model
4. Store them in Qdrant under the collection `gpt_db`

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Qdrant

loader = PyPDFLoader("data.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False}
)

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url="http://localhost:6333",
    collection_name="gpt_db"
)
```

---

## ❓ Ask Questions

After indexing, you can query with natural language:

```python
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333", prefer_grpc=False)

db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name="gpt_db"
)

query = "What are some of the limitations?"
results = db.similarity_search_with_score(query=query, k=5)

for doc, score in results:
    print({"score": score, "content": doc.page_content})
```

---

## 🧪 Example Output

```json
{
  "score": 0.12,
  "content": "One limitation is the reliance on... "
}
```

---

## 📂 Project Structure

```
├── data.pdf
├── main.py           # Your main script
├── README.md         # This file
```

---

## 🛠️ Notes

- You can use `device="cuda"` in `model_kwargs` for GPU acceleration.
- The embedding model does **not normalize embeddings** by default. Change `normalize_embeddings` if needed.

---

## 📃 License

MIT

---

## 🧑‍💻 Author

Crafted with ❤️ by [Victor]
