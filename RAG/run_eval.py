from sentence_window_pipeline import get_sentence_window_query_engine
from auto_merging_pipeline import get_auto_merging_query_engine
from trulens.providers.litellm import LiteLLM

query_engine = get_auto_merging_query_engine()
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

samples = []
for item in golden_set:
    response = query_engine.query(item["question"])
    context_str = "\n".join([node.get_content()
                            for node in response.source_nodes])

    groundedness_score, _ = provider.groundedness_measure_with_cot_reasons(
        source=context_str, statement=str(response))
    relevance_score, _ = provider.relevance_with_cot_reasons(
        prompt=item["question"], response=str(response))
    context_relevance_score, _ = provider.context_relevance_with_cot_reasons(
        question=item["question"], context=context_str)

    samples.append({
        "question": item["question"],
        "groundedness": groundedness_score,
        "answer_relevance": relevance_score,
        "context_relevance": context_relevance_score,
    })
    print(f"Done: {item['question'][:60]}...")
    print(
        f"  Groundedness: {groundedness_score:.2f} | Answer Rel: {relevance_score:.2f} | Context Rel: {context_relevance_score:.2f}")

avg_groundedness = sum(s["groundedness"] for s in samples) / len(samples)
avg_answer_rel = sum(s["answer_relevance"] for s in samples) / len(samples)
avg_context_rel = sum(s["context_relevance"] for s in samples) / len(samples)

print(
    f"\nAVERAGES — Groundedness: {avg_groundedness:.2f} | Answer Relevance: {avg_answer_rel:.2f} | Context Relevance: {avg_context_rel:.2f}")
