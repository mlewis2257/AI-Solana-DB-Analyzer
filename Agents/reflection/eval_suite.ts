import { reflectionGraph } from "./langgraph_reflection_agent.js";

interface EvalCase {
  question: string;
  expectedTools: string[];
}

const evalSet: EvalCase[] = [
  { question: "What is 156 divided by 12?", expectedTools: ["calculator"] },
  {
    question: "What's the average pnl_pct for closed Strategy B trades?",
    expectedTools: ["query_solana_trades"],
  },
  {
    question: "What's the current price of Solana?",
    expectedTools: ["get_crypto_price"],
  },
  {
    question: "Who won the most recent Super Bowl?",
    expectedTools: ["tavily_search"],
  },
  {
    question: "Compare Strategy A and B average pnl_pct on closed trades.",
    expectedTools: ["query_solana_trades"],
  },
  {
    question: "What are the top 5 trades by peak_multiplier?",
    expectedTools: ["query_solana_trades"],
  },
  {
    question: "What's 847 times 23, then divided by 3?",
    expectedTools: ["calculator"],
  },
  {
    question: "How many trades exited via trail_stop for Strategy A?",
    expectedTools: ["query_solana_trades"],
  },
  {
    question:
      "Search for the current price of Bitcoin and tell me what 0.1 BTC is worth.",
    expectedTools: ["get_crypto_price", "calculator"],
  },
  {
    question:
      "What's the win rate for profit_floor exits, broken down by strategy?",
    expectedTools: ["query_solana_trades"],
  },
  {
    question: "What's the 3 worst trades by pnl_pct for Strategy B?",
    expectedTools: ["query_solana_trades"],
  },
  {
    question: "Is there a vip_tier that outperforms untiered trades?",
    expectedTools: ["query_solana_trades"],
  },
  { question: "What's the capital of France?", expectedTools: [] },
  {
    question: "Explain what a stop-loss is, no need to look anything up.",
    expectedTools: [],
  },
  {
    question: "What's the current price of Ethereum and Bitcoin combined?",
    expectedTools: ["get_crypto_price"],
  },
];

async function runEvalSuite() {
  let correctCount = 0;
  const results: any[] = [];

  for (const item of evalSet) {
    const config = {
      configurable: { thread_id: `eval-${Date.now()}-${Math.random()}` },
    };
    const result = await reflectionGraph.invoke(
      {
        messages: [{ role: "user", content: item.question }],
      },
      config,
    );
    const toolsUsed = new Set(
      result.messages
        .filter((m: any) => m.tool_calls?.length)
        .flatMap((m: any) => m.tool_calls?.map((tc: any) => tc.name)),
    );

    const passed =
      item.expectedTools.every((t) => toolsUsed.has(t)) &&
      (item.expectedTools.length > 0 || toolsUsed.size === 0);

    if (passed) correctCount++;
    results.push({
      question: item.question,
      expected: item.expectedTools,
      actual: [...toolsUsed],
      passed,
      revisions: result.revisionCount,
    });

    console.log(`${passed ? "✅" : "❌"} ${item.question.slice(0, 60)}`);
    console.log(
      `   expected: [${item.expectedTools.join(", ")}] | actual: [${[...toolsUsed].join(", ")}] | revisions: ${result.revisionCount}`,
    );
  }
  const accuracy = (correctCount / evalSet.length) * 100;
  console.log(
    `\nTool-selection accuracy: ${correctCount}/${evalSet.length} = ${accuracy.toFixed(1)}%`,
  );
  console.log(
    `Total revisions triggered across all questions: ${results.reduce((sum, r) => sum + r.revisions, 0)}`,
  );
}

runEvalSuite();
