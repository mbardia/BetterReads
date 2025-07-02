import pandas as pd
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient

# 1. Load dataset
df = pd.read_csv("BooksDatasetClean.csv")

# 2. Keep relevant columns, drop rows missing essential info
df = df[["Title", "Authors", "Category", "Description"]]
df.dropna(subset=["Title", "Authors", "Category", "Description"], inplace=True)

# 3. Clean text
def clean_text(text):
    text = str(text).encode("ascii", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text.lower().strip()

df["Description_Clean"] = df["Description"].apply(clean_text)

# 4. Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
print("🔍 Generating embeddings...")
descriptions = df["Description_Clean"].tolist()
embeddings = model.encode(descriptions, convert_to_tensor=False)
df["Embedding"] = embeddings.tolist()

# 5. Initialize ChromaDB client correctly using Settings
client = PersistentClient(path="./chroma_db")

# 6. Create or load collection
collection = client.get_or_create_collection(name="books")

# 7. Add data in batches
batch_size = 500
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    ids = [f"{i+j}" for j in range(len(batch))]
    documents = batch["Description_Clean"].tolist()
    embeddings = batch["Embedding"].tolist()
    metadatas = batch[["Title", "Authors", "Category"]].to_dict(orient="records")

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

# 8. Save to disk
client.persist()
print("✅ ChromaDB vector store created and saved in ./chroma_db")
