import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, LLMRerank
# from ragas import evaluate, EvaluationDataset
# from ragas.llms import LangChainLLMWrapper
# from ragas.metrics import Fathfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
# from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from trulens.providers.litellm import LiteLLM
from trulens.core import Metric, Selector
from langchain_anthropic import ChatAnthropic

load_dotenv()
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")


Settings.llm = Anthropic(model="claude-sonnet-4-6")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Sentence Window Parser splits into individual sentences and attaches each a window
# of surrounding sentences to each one's metadata

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text"
)

documents = SimpleDirectoryReader(input_dir="data").load_data()
# print(f"Loaded: {len(documents)} documents")
nodes = node_parser.get_nodes_from_documents(documents)
print(f"Split {len(documents)} documents into {len(nodes)} sentence-level nodes.")

if not os.path.exists("./sentence_window_index"):
    sentence_index = VectorStoreIndex(nodes)
    sentence_index.storage_context.persist(
        persist_dir="./sentence_window_index")
else:
    storage_context = StorageContext.from_defaults(
        persist_dir="./sentence_window_index")
    sentence_index = load_index_from_storage(storage_context)

# Post-processor: swaps the single retrieved sentence back out for its full window
post_proc = MetadataReplacementPostProcessor(target_metadata_key="window")

# LLM-based re-ranker: retrieve 6 candidates, ask Claude to pick and reorder the best 2
rerank = LLMRerank(choice_batch_size=5, top_n=2)
query_engine = sentence_index.as_query_engine(
    similarity_top_k=6, node_postprocessors=[post_proc, rerank])

# Providers for LiteLLM pointed at Claude
provider = LiteLLM(model_engine="claude-sonnet-4-6")

golden_set = [
    {"question": "What is the role of the hippocampus in memory?"},
    {"question": "What happens during an action potential?"},
    {"question": "What is a synapse and what happens there?"},
    {"question": "What is neuroplasticity?"},
    {"question": "What are the main functions of the cerebral cortex?"},
    {"question": "What is the basic structure of a neuron?"},
    {"question": "How does damage to the hippocampus affect memory differently than damage to the cortex?"},
    {"question": "What role do neurotransmitters play at the synapse?"},
    {"question": "Is neuroplasticity only relevant to childhood development?"},
    {"question": "What ion channels are involved in generating an action potential?"},
    {"question": "Can the hippocampus repair itself after damage?"},
    {"question": "What distinguishes the cerebral cortex from deeper brain structures like the hippocampus?"},
]
# Run every question in the golden set through the RAG pipeline and collect responses

samples = []
for item in golden_set:
    response = query_engine.query(item["question"])
    context_str = "\n".join([node.get_content()
                            for node in response.source_nodes])
    groundness_score, groundness_reason = provider.groundedness_measure_with_cot_reasons(
        source=context_str, statement=str(response)
    )
    relevance_score, relevance_reason = provider.relevance_with_cot_reasons(
        prompt=item["question"], response=str(response)
    )
    context_relevance_score, context_relevance_reason = provider.context_relevance_with_cot_reasons(
        question=item["question"], context=context_str
    )

    samples.append({
        "user_input": item["question"],
        "groundedness": groundness_score,
        "answer_relevance": relevance_score,
        "context_relevance": context_relevance_score
    })
    print(f"Done: {item['question'][:60]}...")
    print(
        f"Groundedness : {groundness_score:.2f} | Answer Rel: {relevance_score:.2f} | Context Rel: {context_relevance_score:.2f}")

avg_groundedness = sum(s["groundedness"] for s in samples) / len(samples)
avg_answer_rel = sum(s["answer_relevance"] for s in samples) / len(samples)
avg_context_rel = sum(s["context_relevance"] for s in samples) / len(samples)

# RAGAS Evaluations
# dataset = EvaluationDataset.from_list(samples)
# ragas_llm = LangChainLLMWrapper(ChatAnthropic(model="claude-sonnet-4-6"))
# ragas_embeddings = LlamaIndexEmbeddingsWrapper(
#     OllamaEmbedding(model_name="nomic-embed-text"))

# result = evaluate(dataset=dataset,
#                   metrics=[Fathfulness(), ResponseRelevancy(),
#                            LLMContextPrecisionWithoutReference(), LLMContextRecall()],
#                   llm=ragas_llm,
#                   embeddings=ragas_embeddings
#                   )
print(
    f"\n AVERAGES - Groundedness: {avg_groundedness:.2f} | Answer_Relevance: {avg_answer_rel:.2f} | Context_Relevance: {avg_context_rel:.2f}")
