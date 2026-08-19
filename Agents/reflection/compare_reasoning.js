import { runAgent } from "./agent.js";
import { Anthropic } from "@anthropic-ai/sdk/client.js";

const anthropic = new Anthropic();

const synthesizeClaude = async (question, gatheredInfo) => {
  const start = Date.now();
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 1024,
    system: `Answer directly and concisely based on the gathered information provided. If there's genuine ambiguity in the data, note it briefly`,
    messages: [
      {
        role: "user",
        content: `Question: ${question}\n\nInformation gathered: ${gatheredInfo}\n\nGive the direct final answer`,
      },
    ],
  });
  const text = response.content.find((b) => b.type === "text")?.text ?? "";
  const elapsedMs = Date.now() - start;
  const inputTokens = response.usage.input_tokens;
  const outputTokens = response.usage.output_tokens;
  const costUsd =
    (inputTokens / 1_000_000) * 3 + (outputTokens / 1_000_000) * 15;
  return { text, elapsedMs, costUsd };
};

const synthesizeLocalDeepSeek = async (question, gatheredInfo) => {
  const start = Date.now();
  const response = await fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "deepseek-r1:8b",
      messages: [
        {
          role: "user",
          content: `Question: ${question}\n\nInformation gathered: ${gatheredInfo}\n\nGive the direct final answer`,
        },
      ],
      stream: false,
    }),
  });
  const data = await response.json();
  const elapsedMs = Date.now() - start;
  return {
    text: data.message?.content ?? "(no response)",
    elapsedMs,
    costUsd: 0,
  };
};

const compare = async (question) => {
  console.log(`\n${"=".repeat(70)}\nQUESTION: ${question}\n${"=".repeat(70)}`);

  console.log("Gathering data (Claude + tools)...");
  const gathered = await runAgent(question);

  console.log("\n--- Claude Sonnet Synthesis ---");
  const claudeResult = await synthesizeClaude(question, gathered.finalText);
  console.log(claudeResult.text);
  console.log(
    `(${claudeResult.elapsedMs}ms, ~$${claudeResult.costUsd.toFixed(5)})`,
  );
  console.log("\n--- DeepSeek-r1:8b Synthesis");
  const deepseekResult = await synthesizeLocalDeepSeek(
    question,
    gathered.finalText,
  );
  console.log(deepseekResult.text);
  console.log(`${deepseekResult.elapsedMs}ms, ~$0 --runs locally}`);

  return { question, claudeResult, deepseekResult };
};

const question = [
  "Compare Strategy A and Strategy B: what's the average pnl_pct for each on closed trades?",
  "Is there a difference in average pnl_pct between trades with a conviction_score above 50 versus below 50?",
];

for (const q of question) {
  await compare(q);
}
