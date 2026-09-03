import { runAgentWithReflection } from "./reflection.js";

const questions = [
  "What are the top 15 trades by pnl_pct across both strategies, in a single ranked table?",
  "Compare average pnl_pct for trades with narrative_tags containing 'meme' vs everything else.",
  "Is there a vip_tier that performs better than others on average pnl_pct?",
  "Give me the 3 worst trades by pnl_pct for Strategy B, with their exit_reason and conviction_score.",
  "What's the win rate for trades exited via profit_floor, broken down by strategy?",
];

for (const q of questions) {
  console.log(`\n${"=".repeat(70)}\nQUESTION:${q}\n${"=".repeat(70)}`);
  const result = await runAgentWithReflection(q);
  console.log("Final:", result.finalText);
  console.log("Genuine Revision:", result.genuineRevision);
  console.log("Critique Note:", result.critiqueNote);
  console.log("Tools Called:", result.toolsCalled);
}
