import { Anthropic } from "@anthropic-ai/sdk/client.js";
import { runSolanaQuery } from "./solana_db.js";
import dotenv from "dotenv";

dotenv.config();
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;
const anthropic = new Anthropic(ANTHROPIC_API_KEY);

console.log(
  "Key loaded:",
  process.env.ANTHROPIC_API_KEY ? "yes" : "NO - undefined",
);

const mockVenues = [
  {
    name: "Velvet Lounge",
    moodTag: "chill",
    capacity: 120,
    bookings: [
      { date: "2026-08-01", partySize: 40 },
      { date: "2026-08-02", partySize: 85 },
    ],
  },
  {
    name: "Neon District",
    moodTag: "high-energy",
    capacity: 300,
    bookings: [
      { date: "2026-08-01", partySize: 210 },
      { date: "2026-08-02", partySize: 300 },
    ],
  },
  {
    name: "The Grove",
    moodTag: "romantic",
    capacity: 60,
    bookings: [{ date: "2026-08-01", partySize: 15 }],
  },
];
// Tool definition: name, description, and a JSON schema for its inputs.
// This description is the ENTIRE interface Claude has for deciding when/how to use it.
const tools = [
  {
    name: "calculator",
    description:
      "Evaluates a basic arithmetic expression (add, subtract, multiply, divide) and returns the numeric result.",
    input_schema: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description: "A math expression to evaluate, e.g. '12 * (4 + 3)'",
        },
      },
      required: ["expression"],
    },
  },
  {
    type: "web_search_20260318",
    name: "web_search",
  },
  {
    name: "query_color_db",
    description:
      "Looks up a COLOR venue by name and returns its mood tag, capacity, and upcoming bookings. Use this for any question about a specific venue's data, occupancy, or scheduled bookings.",
    input_schema: {
      type: "object",
      properties: {
        venue_name: {
          type: "string",
          description: "The name of the venue to look up, e.g. 'Neon District'",
        },
      },
      required: ["venue_name"],
    },
  },
  {
    name: "query_solana_trades",
    description:
      "Queries the trading bot's historical trade outcomes. For 'how many/average/total' questions, set aggregate=true. For 'highest/lowest/top N by [field]' questions, you MUST set order_by to that field — do not rely on sorting the results yourself. conviction_score ranges from 0-100 (not 0-10), average ~49.5 -- verify assumptions about any field's scale against real data rather than guessing. If a filter returns zero results, say so plainly rather than substituting a different dataset. if asked about live/open/active trades, first query with that status filter — do not assume only closed trades exist and if there are no open trades state this plainly then run the analysis on on closed trades instead.",
    input_schema: {
      type: "object",
      properties: {
        symbol: {
          type: "string",
          description: "Token symbol",
        },
        status: {
          type: "string",
          description: "Trade status, e.g. 'closed' or 'open'",
        },
        is_strategy_b: {
          type: "boolean",
          description:
            "Filter to only Strategy B (true) or Strategy A (false) trades",
        },
        exit_reason: {
          type: "string",
          description: "e.g. 'stop_loss', 'take_profit', 'trail_stop'",
        },
        min_pnl_pct: {
          type: "number",
          description: "Minimum profit/loss percentage",
        },
        max_pnl_pct: {
          type: "number",
          description: "Maximum profit/loss percentage",
        },
        min_conviction_score: {
          type: "number",
          description: "Minimum conviction score for potential trades",
        },
        max_conviction_score: {
          type: "number",
          description: "Maximum conviction score for potential trades",
        },
        order_by: {
          type: "string",
          enum: [
            "exit_time",
            "entry_time",
            "pnl_pct",
            "peak_multiplier",
            "conviction_score",
          ],
          description: "Field to sort by, default exit_time",
        },
        vip_tier: { type: "string" },
        limit: {
          type: "number",
          description: "Max rows to return, 1-100, default 20",
        },
        aggregate: {
          type: "boolean",
          description:
            "Set true for count/average/sum/min/max questions instead of individual rows",
        },
      },
      required: [],
    },
  },
];

// The actual function that runs when Claude calls the "calculator" tool.
// This is real code YOU wrote -- Claude never executes anything itself,
// it just asks you to, and you decide how.
const runColorDBQuery = (input) => {
  const venue = mockVenues.find(
    (v) => v.name.toLowerCase() === input.venue_name.toLowerCase(),
  );
  if (!venue) {
    return `No venue found matching "${input.venue}".`;
  }
  const totalBooked = venue.bookings.reduce((sum, b) => sum + b.partySize, 0);
  return JSON.stringify({
    name: venue.name,
    moodTag: venue.moodTag,
    capacity: venue.capacity,
    bookings: venue.bookings,
    totalBookedThisWeekend: totalBooked,
  });
};

const runCalculator = (input) => {
  try {
    const results = Function(`"use strict"; return (${input.expression})`)();
    return String(results);
  } catch (error) {
    return `Error evaluating message: ${error.message}`;
  }
};

const TIME_SENSITIVE_KEYWORDS =
  /\b(today|current|latest|now|this week|score|weather|price)\b/i;

export const runAgent = async (
  userQuery,
  onEvent = () => {},
  priorState = null,
) => {
  const messages = priorState
    ? [...priorState.messages, { role: "user", content: userQuery }]
    : [{ role: "user", content: userQuery }];
  const toolsCalled = priorState ? [...priorState.toolsCalled] : [];
  let containerId = priorState?.containerId || null;
  // const forceSearch = TIME_SENSITIVE_KEYWORDS.test(userQuery);

  while (true) {
    let response;
    try {
      response = await anthropic.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 4096,
        system: `Today's date is ${new Date().toISOString().split("T")[0]}. Always use web_search for current events, sports results, dates, weather, or prices — never answer these from memory.

When giving a final answer: be direct and concise. Do not use markdown headers, multiple sections, or emoji unless the user explicitly asks for a detailed breakdown. A simple factual question deserves a short, plain-prose answer — a sentence or two, not a formatted report.`,
        tools,
        tool_choice: { type: "auto" },
        ...(containerId ? { container: containerId } : {}),
        messages,
      });
    } catch (error) {
      console.error("API error:", error.message);
      onEvent({ type: "error", message: error.message });
      return { finalText: "", toolsCalled, messages, containerId };
    }

    if (response.container?.id) {
      containerId = response.container.id;
    }

    // Add claude's response to the conversation history

    messages.push({ role: "assistant", content: response.content });

    if (response.stop_reason !== "tool_use") {
      const finalText =
        response.content.find((b) => b.type === "text")?.text ?? "";
      onEvent({ type: "final_answer", text: finalText });
      return { finalText, toolsCalled, messages, containerId };
    }

    // Claude wants to use one or more tools -- find and run each one
    const toolResult = [];
    for (const block of response.content) {
      if (block.type === "server_tool_use") {
        toolsCalled.push(block.name);
        onEvent({
          type: "tool_call",
          tool: block.name,
          input: block.input,
          serverExecuted: true,
        });
      } else if (block.type === "web_search_tool_result") {
        onEvent({
          type: "tool_call",
          tool: "web_search",
          resultCount: block.content.length,
        });
      } else if (block.type === "tool_use") {
        toolsCalled.push(block.name);
        onEvent({
          type: "tool_call",
          tool: block.name,
          input: block.input,
          serverExecuted: false,
        });
        let result;
        if (block.name === "calculator") {
          result = runCalculator(block.input);
        } else if (block.name === "query_solana_trades") {
          result = await runSolanaQuery(block.input);
        } else if (block.name === "query_color_db") {
          result = runColorDBQuery(block.input);
        } else {
          result = `Unknown tool: ${block.name}`;
        }
        onEvent({ type: "tool_result", tool: block.name, result });
        toolResult.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }
    if (toolResult.length > 0) {
      messages.push({ role: "user", content: toolResult });
    }
  }
};
