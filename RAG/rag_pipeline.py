import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import OpenAIEmbedding
from llama_index.core import Settings

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Global config - replaces the old ServiceContext entirely
Settings.llm = Ollama(model="llama-3.1", request_timeout=120.0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Step 1: Load the documents and preare docs
# Load Docs
documents = SimpleDirectoryReader(
    input_files=["./Stock_Market_WIZARDS_Interviews_with_Ame.pdf"]
).load_data()
print(f"Loaded {len(documents)} documents")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What is this document about?")

print(str(response))
