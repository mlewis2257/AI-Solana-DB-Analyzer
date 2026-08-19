import { runAgentWithReflection } from "./reflection.js";

const result = await runAgentWithReflection(
  "Search for the current population of Tokyo, and tell me what it would be if it dropped 12%",
);

if (result.genuineRevision) {
  console.log("LOW CONFIDENCE:", result.critiqueNote);
  console.log("Answer (verify before trusting):", result.finalText);
} else {
  console.log("Answer:", result.finalText);
}

console.log("\nFinal:", result.finalText);
console.log("Revisions Needed:", result.needsRevision);
console.log("Tools Used:", result.toolsCalled);
console.log("Critique note:", result.critiqueNote);
