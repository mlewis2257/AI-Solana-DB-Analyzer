import {
  StateGraph,
  MemorySaver,
  Annotation,
  InMemoryStore,
  type BaseStore,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { toolsCondition, ToolNode } from "@langchain/langgraph/prebuilt";
import {
  synthesizeNode,
  critiqueNode,
  ReflectionState,
  latestQuestion,
} from "./langgraph_reflection_agent.js";
import { tools, model } from "./langgraph_agent.js";
import { SystemMessage } from "@langchain/core/messages";
import { createHash } from "node:crypto";

const memoryStore = new InMemoryStore();
const NAMESPACE_INSIGHTS = ["solana-agent", "data_insights"];
const NAMESPACE_PREF = ["solana-agent", "preferences"];
const NAMESPACE_LOG = ["solana-agent", "query_log"];

const MemoryState = Annotation.Root({
  ...ReflectionState.spec,
});

const RUN_RESET = {
  critiqueFeedback: null,
  revisionCount: 0,
  draftAnswer: "",
};

async function recallNode(
  state: typeof MemoryState.State,
  config: LangGraphRunnableConfig,
) {
  const store = config.store;
  if (!store) {
    console.error("recallNode: no store on config, compile with { store }");
    return { ...RUN_RESET, recalledContext: "" };
  }

  try {
    const preferences = await recallFrom(store, NAMESPACE_PREF, "preference");
    const insights = await recallFrom(store, NAMESPACE_INSIGHTS, "insight");
    const past_queries = await recallFrom(store, NAMESPACE_LOG, "past_query");

    const sections: string[] = [];

    if (preferences.length) {
      sections.push(
        "User preferences (follow these):\n" +
          preferences.map((p) => `- ${p}`).join("\n"),
      );
    }
    if (insights.length) {
      sections.push(
        "User Insights (hints only, verify with tools before relying on them):\n" +
          insights.map((p) => `- ${p}`).join("\n"),
      );
    }
    if (past_queries.length) {
      sections.push(
        "User past queries (questions this user has asked before):\n" +
          past_queries.map((p) => `- ${p}`).join("\n"),
      );
    }
    return { ...RUN_RESET, recalledContext: sections.join("\n\n") };
  } catch (error) {
    console.error("recallNode failed:", error);
    return { ...RUN_RESET, recalledContext: "" };
  }
}

async function recallFrom(
  store: BaseStore,
  namespace: string[],
  field: string,
): Promise<string[]> {
  const items = await store.search(namespace, { limit: 50 });
  return items
    .map((item) => item.value[field])
    .filter((v): v is string => typeof v === "string" && v.length > 0);
}

async function memoryAssistantNode(state: typeof MemoryState.State) {
  const messages = state.recalledContext
    ? [new SystemMessage(state.recalledContext), ...state.messages]
    : state.messages;

  const response = await model.invoke(messages);

  return { messages: [response] };
}

async function rememberNode(
  state: typeof MemoryState.State,
  config: LangGraphRunnableConfig,
) {
  const store = config.store;
  if (!store) {
    console.error("rememberNode: no store on config");
    return {};
  }
  try {
    const question = latestQuestion(state.messages);
    if (!question) return {};

    const normalized = question.trim().toLowerCase();
    const key = createHash("sha256")
      .update(normalized)
      .digest("hex")
      .slice(0, 16);

    await store.put(NAMESPACE_LOG, key, { past_query: question });
  } catch (error) {
    console.error("Failed to log query", error);
  }
  return {};
}

function routeAfterCritique(
  state: typeof MemoryState.State,
): "synthesize" | "remember" | "__end__" {
  if (state.critiqueFeedback && state.revisionCount < 2) {
    return "synthesize";
  } else if (!state.critiqueFeedback) {
    return "remember";
  }
  return "__end__";
}

const builder = new StateGraph(MemoryState)
  .addNode("recall", recallNode)
  .addNode("assistant", memoryAssistantNode)
  .addNode("tools", new ToolNode(tools))
  .addNode("synthesize", synthesizeNode)
  .addNode("critique", critiqueNode)
  .addNode("remember", rememberNode)
  .addEdge("__start__", "recall")
  .addEdge("recall", "assistant")
  .addConditionalEdges("assistant", toolsCondition, {
    tools: "tools",
    __end__: "synthesize",
  })
  .addEdge("tools", "assistant")
  .addEdge("synthesize", "critique")
  .addConditionalEdges("critique", routeAfterCritique, {
    synthesize: "synthesize",
    remember: "remember",
    __end__: "__end__",
  })
  .addEdge("remember", "__end__");
const checkpointer = new MemorySaver();
export const memoryGraph = builder.compile({
  checkpointer,
  store: memoryStore,
});

async function ask(question: string, threadId: string) {
  console.log(`\n===== ${threadId}: ${question} =====`);
  const stream = await memoryGraph.stream(
    {
      messages: [
        {
          role: "user",
          content: question,
        },
      ],
    },
    { configurable: { thread_id: threadId }, streamMode: "updates" },
  );
  for await (const update of stream) {
    for (const [node, output] of Object.entries(update)) {
      console.log("ran:", node, "| wrote", Object.keys(output ?? {}));
      if (node === "synthesize") console.log("\n", output, "\n");
      if (node === "recall" && output && "recalledContext" in output) {
        console.log(output.recalledContext || "(empty)");
      }
    }
  }
}

async function main() {
  await memoryStore.put(NAMESPACE_PREF, "answer-format", {
    preference: "Present lists of trades as a markdown table",
  });
  await memoryStore.put(NAMESPACE_INSIGHTS, "seed-insight", {
    insight: "Trading size for live bot is .05 SOL per trade",
  });

  await ask(
    "What are the top 10 highest PnL trades from the live trading bot?",
    "mem-1",
  );
  await ask("What is the average PnL % for live trades?", "mem-2");
}
//   const result = await memoryGraph.invoke(
//     {
//       messages: [
//         {
//           role: "user",
//           content:
//             "What are the top 10 highest PnL trades from the live trading bot, ranked together correctly?",
//         },
//       ],
//     },
//     { configurable: { thread_id: "memory_test_1" } },
//   );

//   console.log("--- recalledContext ---");
//   console.log(result.recalledContext || "(empty)");
//   console.log("\n--- final answer ---");
//   console.log(result.draftAnswer);
//   console.log("\nrevisions:", result.revisionCount);
// }

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
