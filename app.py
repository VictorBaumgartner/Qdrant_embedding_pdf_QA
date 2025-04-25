from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient 
from qdrant_client.models import Distance, VectorParams

#load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings (
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

url = "http://localhost:6333"
collection_name = "gpt_db"

client = QdrantClient(
    url = url, 
    prefer_grpc = False
)
 
print(client)
print("###############")

# Create collection if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.embedding_size, distance=Distance.COSINE)
    )

# Load existing collection
db = Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

print(db)
print("##################")

query = "What are some of the limitations ?"
docs = db.similarity_search_with_score(query=query, k=5)

for i in docs: 
    doc, score = i
    print({"score": score, "content": doc.page_content, "metadata": doc.metadata})
