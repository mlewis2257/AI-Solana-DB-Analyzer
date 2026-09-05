import {
  StateGraph,
  MessagesAnnotation,
  MemorySaver,
  messagesDeltaReducer,
} from "@langchain/langgraph";
import { ToolNode, toolsCondition } from "@langchain/langgraph/prebuilt";
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { runSolanaQuery } from "./solana_db.js";
import { TavilySearch } from "@langchain/tavily";

const calculatorSchema = z.object({
  expression: z
    .string()
    .describe("A math expression to evaluate, e.g. '12 * (4 + 3)'"),
});

type CalculatorInput = z.infer<typeof calculatorSchema>;

const calculatorTool = tool(
  async ({ expression }: CalculatorInput) => {
    try {
      const results = Function(`"use strict"; return (${expression})`)();
      return String(results);
    } catch (error) {
      return `Error evaluating message: ${(error as Error).message}`;
    }
  },
  {
    name: "calculator",
    description:
      "Evaluates a basic arithmetic expression and returns the numeric result.",
    schema: calculatorSchema,
  },
);

const solanaQuerySchema = z.object({
  symbol: z.string().optional(),
  status: z.string().optional(),
  is_strategy_b: z.boolean().optional(),
  exit_reason: z.string().optional(),
  min_pnl_pct: z.number().optional(),
  max_pnl_pct: z.number().optional(),
  min_conviction_score: z.number().optional(),
  max_conviction_score: z.number().optional(),
  order_by: z
    .enum([
      "exit_time",
      "entry_time",
      "pnl_pct",
      "peak_multiplier",
      "conviction_score",
    ])
    .optional(),
  order_direction: z.enum(["asc", "desc"]).optional(),
  vip_tier: z.string().optional(),
  limit: z.number().optional(),
  aggregate: z.boolean().optional(),
});

type SolanaQueryInput = z.infer<typeof solanaQuerySchema>;

const solanaQueryTool = tool(
  async (input: SolanaQueryInput) => {
    try {
      return await runSolanaQuery(input);
    } catch (error) {
      return `Error querying from the input: ${(error as Error).message}`;
    }
  },
  {
    name: "query_solana_trades",
    description:
      "Queries the trading bot's historical trade outcomes. For 'how many/average/total' questions, set aggregate=true. For 'highest/lowest/top N by [field]' questions, set order_by AND order_direction. conviction_score ranges 0-100.",
    schema: solanaQuerySchema,
  },
);

const cryptoPriceSchema = z.object({
  coins: z
    .array(z.string())
    .describe(
      "CoinGecko coin IDs, e.g. ['bitcoin', 'solana', 'ethereum', 'binancecoin']",
    ),
});

type CryptoPriceInput = z.infer<typeof cryptoPriceSchema>;

const cryptoPriceTool = tool(
  async ({ coins }: CryptoPriceInput) => {
    try {
      const ids = coins.join(",");
      const response = await fetch(
        `https://api.coingecko.com/api/v3/simple/price?ids=${ids}&vs_currencies=usd`,
      );
      const data = await response.json();
      return JSON.stringify(data);
    } catch (error) {
      return `Error fetching prices: ${(error as Error).message}`;
    }
  },
  {
    name: "get_crypto_price",
    description:
      "Gets the current real-time USD price for one or more cryptocurrencies. Use this instead of web_search for any question about current crypto prices — web search returns stale article data, not live prices. Common coin IDs: bitcoin, ethereum, solana, binancecoin.",
    schema: cryptoPriceSchema,
  },
);

const webSearchTool = new TavilySearch({
  maxResults: 5,
});

const tools = [
  calculatorTool,
  solanaQueryTool,
  webSearchTool as any,
  cryptoPriceTool,
];
const model = new ChatAnthropic({ model: "claude-sonnet-4-6" }).bindTools(
  tools,
);

async function assistantNode(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
}

const builder = new StateGraph(MessagesAnnotation)
  .addNode("assistant", assistantNode)
  .addNode("tools", new ToolNode(tools))
  .addEdge("__start__", "assistant")
  .addConditionalEdges("assistant", toolsCondition)
  .addEdge("tools", "assistant");

const checkpointer = new MemorySaver();
export const graph = builder.compile({ checkpointer });

async function main() {
  const result = await graph.invoke(
    {
      messages: [
        {
          role: "user",
          content:
            "Compare Strategy A and Strategy B: what's the average pnl_pct for each on closed trades?",
        },
      ],
    },
    {
      configurable: { thread_id: "test-2" },
    },
  );
  const lastMessage = result.messages[result.messages.length - 1];
  console.log(`Final Answer:`, lastMessage?.content);
}

async function main_search() {
  const result = await graph.invoke(
    {
      messages: [
        {
          role: "user",
          content:
            "What is the price of Bitcoin, Solana, Binance and Ethereum as of today?",
        },
      ],
    },
    {
      configurable: { thread_id: "test-2" },
    },
  );
  const lastMessage = result.messages[result.messages.length - 1];
  console.log("Final Message:", lastMessage?.content);
}

export { tools, model, assistantNode };

if (import.meta.url === `file://${process.argv[1]}`) {
  main_search();
}
