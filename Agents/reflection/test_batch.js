import { runAgentWithReflection } from "./reflection.js";

const questions = [
  //   "Compare Strategy A and Strategy B: what's the average pnl_pct for each on closed trades?",
  //   "What's the win rate (percentage of trades with positive pnl_pct) for closed trades that exited via take_profit?",
  //   "Show me the 5 most recent closed trades with the highest peak_multiplier, and tell me what their exit_reason was.",
  //   "Is there a difference in average pnl_pct between trades with a conviction_score above 7 versus below 7?",
  "Analyze all the win rate on the Live trades and tell me what the average pnl has been?",
];

for (const q of questions) {
  console.log(`\n${"=".repeat(70)}\nQUESTION:${q}\n${"=".repeat(70)}`);
  const result = await runAgentWithReflection(q);
  console.log("Final:", result.finalText);
  console.log("Genuine Revision:", result.genuineRevision);
  console.log("Critique Note:", result.critiqueNote);
  console.log("Tools Called:", result.toolsCalled);
}
