import {
  StateGraph,
  MemorySaver,
  Annotation,
  type BaseStore,
  type LangGraphRunnableConfig,
} from "@langchain/langgraph";
import { PostgresStore } from "@langchain/langgraph-checkpoint-postgres/store";
import { toolsCondition, ToolNode } from "@langchain/langgraph/prebuilt";
import {
  synthesizeNode,
  critiqueNode,
  ReflectionState,
  latestQuestion,
  plainModel,
  toolResultsText,
} from "./langgraph_reflection_agent.js";
import { tools, model } from "./langgraph_agent.js";
import { SystemMessage } from "@langchain/core/messages";
import { createHash } from "node:crypto";
import { z } from "zod";

const MEMORY_DB_URL = process.env.MEMORY_DB_URL;
if (!MEMORY_DB_URL) {
  throw new Error(
    "MEMORY_DB_URL is not set. Add it to .env and the compose file.",
  );
}

const memoryStore = PostgresStore.fromConnString(MEMORY_DB_URL);
const NAMESPACE_INSIGHTS = ["solana-agent", "data-insights"];
const NAMESPACE_PREF = ["solana-agent", "preferences"];
const NAMESPACE_LOG = ["solana-agent", "query-log"];

const MemoryState = Annotation.Root({
  ...ReflectionState.spec,
});

const RUN_RESET = {
  critiqueFeedback: null,
  revisionCount: 0,
  draftAnswer: "",
  approved: false,
};

const insightSchema = z.object({
  insight: z
    .string()
    .nullable()
    .describe(
      "One durable pattern from the data, or null if there is none worth remembering.",
    ),
});

const insightModel = plainModel.withStructuredOutput(insightSchema);

const verfiySchema = z.object({
  reasoning: z
    .string()
    .describe(
      "Check each claim in the insight against the tool results, briefly.",
    ),
  supported: z
    .boolean()
    .describe(
      "true only if every claim is directly supported, with the stated scope.",
    ),
});

const verifyModel = plainModel.withStructuredOutput(verfiySchema);

function hashKey(text: string): string {
  return createHash("sha256")
    .update(text.trim().toLowerCase())
    .digest("hex")
    .slice(0, 16);
}

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
  const question = latestQuestion(state.messages);
  if (!question) return {};
  try {
    await store.put(NAMESPACE_LOG, hashKey(question), { past_query: question });
  } catch (error) {
    console.error("Failed to log query", error);
  }
  try {
    const prompt = `You decide whether an answered question revealed something worth remembering for future sessions.

Question: ${question}

Approved answer: ${state.draftAnswer}

Raw tool results (the only evidence you may use):
${toolResultsText(state.messages)}

Return one insight only if the tool results show a pattern or comparison likely to stay true for weeks, such as how one group of trades performs relative to another.

Rules:
- Use only the raw tool results as evidence, never the answer's wording alone.
- One sentence, under 25 words. Exactly one scope: live trades or paper trades, never both.
- No numbers at all.
- No advice, opinions, or recommendations.
- If the question was a simple lookup, the answer is a single number, or the tools failed, return null.

Most answers should return null.`;

    const { insight } = await insightModel.invoke(prompt);

    if (!insight || !insight.trim()) {
      console.log("rememberNode: no durable insight");
      return {};
    }

    const check = await verifyModel.invoke(
      `You are checking a proposed memory before it is saved permanently.

Proposed insight: ${insight}

Raw tool results (the only evidence):
${toolResultsText(state.messages)}

Check every claim in the insight against the tool results. If any part is not directly supported, overgeneralizes (for example "consistently" or "across both" when the data shows an exception), or mixes live and paper scope, it is not supported.`,
    );

    if (!check.supported) {
      console.log(
        "rememberNode: insight REJECTED, not fully supported by tool results",
        insight,
      );
      console.log("reasoning:", check.reasoning);
      return {};
    }

    await store.put(NAMESPACE_INSIGHTS, hashKey(insight), {
      insight,
      recorded_at: new Date().toISOString(),
      source_query: question,
    });
    console.log("rememberNode: stored insight:", insight);
  } catch (error) {
    console.error("rememberNode: insight extraction failed", error);
  }

  return {};
}

function routeAfterCritique(
  state: typeof MemoryState.State,
): "synthesize" | "remember" | "__end__" {
  if (state.critiqueFeedback && state.revisionCount < 2) {
    return "synthesize";
  } else if (state.approved) {
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
  const [question, threadId] = process.argv.slice(2);
  if (!question || !threadId) {
    console.error(
      'Usage: npx tsx langgraph_memory_agent.ts "<question>" <thread-id>',
    );
    process.exit(1);
  }
  try {
    await memoryStore.setup();

    await memoryStore.put(NAMESPACE_PREF, "answer-format", {
      preference: "Present lists of trades as a markdown table",
    });
    await memoryStore.put(NAMESPACE_INSIGHTS, "seed-insight", {
      insight: "Trading size for live bot is .05 SOL per trade",
    });
    await ask(question, threadId);
  } finally {
    await memoryStore.stop();
  }
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
