import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { runSolanaQuery } from "./solana_db.js";

const server = new McpServer({
  name: "solana-trade-analyst",
  version: "1.0.0",
});

server.registerTool(
  "query_solana_trades",

  {
    description:
      "Queries the trading bot's historical trade outcomes. For 'how many/average/total' questions, set aggregate=true. For 'highest/lowest/top N by [field]' questions, set order_by AND order_direction. conviction_score ranges 0-100.",
    inputSchema: {
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
    },
  },
  async (input) => {
    const result = await runSolanaQuery(input);
    return {
      content: [{ type: "text", text: result }],
    };
  },
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((error) => {
  console.error("Error starting MCP server:", error);
  process.exit(1);
});
