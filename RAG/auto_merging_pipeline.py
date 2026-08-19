import os
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

Settings.llm = Anthropic(model="claude-sonnet-4-6")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


def get_auto_merging_query_engine(persist_dir="./auto_merging_index", data_dir="data"):
    # 3-Layer Heirarchical Structure: 128-token leaves, 512-token parents, 2048-token grandparents
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[2048, 512, 128])

    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
        all_nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(all_nodes)
        print(
            f"Parsed {len(documents)} documents into {len(all_nodes)} total nodes ({len(leaf_nodes)} leaf nodes).")

        docstore = SimpleDocumentStore()
        docstore.add_documents(all_nodes)
        storage_context = StorageContext.from_defaults(docstore=docstore)

        auto_merging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context)
        auto_merging_index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        auto_merging_index = load_index_from_storage(storage_context)

    base_retriever = auto_merging_index.as_retriever(similarity_top_k=12)

    retriever = AutoMergingRetriever(
        base_retriever, auto_merging_index.storage_context, verbose=True)
    rerank = LLMRerank(choice_batch_size=5, top_n=6)

    return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[rerank])
