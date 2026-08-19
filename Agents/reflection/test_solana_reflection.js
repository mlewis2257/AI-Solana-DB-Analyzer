import { runAgentWithReflection } from "./reflection.js";

const result = await runAgentWithReflection(
  "How many closed Strategy B trades exited via trail_stop, and what was their average pnl_pct?",
);

console.log("\nFinal:", result.finalText);
console.log("Genuine revision:", result.genuineRevision);
console.log("Critique note:", result.critiqueNote);
console.log("Tools used:", result.toolsCalled);
