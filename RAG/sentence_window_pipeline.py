import os
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, LLMRerank
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")

Settings.llm = Anthropic(model="claude-sonnet-4-6")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


def get_sentence_window_query_engine(persist_dir="./sentence_window_index", data_dir="data"):
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=5,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    if not os.path.exists(persist_dir):
        documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
        nodes = node_parser.get_nodes_from_documents(documents)
        print(
            f"Split {len(documents)} documents into {len(nodes)} sentence-level nodes.")
        sentence_index = VectorStoreIndex(nodes)
        sentence_index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        sentence_index = load_index_from_storage(storage_context)

    post_proc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = LLMRerank(choice_batch_size=5, top_n=2)

    return sentence_index.as_query_engine(
        similarity_top_k=6,
        node_postprocessors=[post_proc, rerank],
    )
