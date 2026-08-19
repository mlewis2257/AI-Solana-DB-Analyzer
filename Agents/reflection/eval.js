import { runAgent } from "./agent.js";

const evalSet = [
  { task: "What is 156 divided by 12?", expectedTools: ["calculator"] },
  {
    task: "What's the capacity of Neon District?",
    expectedTools: ["query_color_db"],
  },
  {
    task: "Who won the most recent Super Bowl?",
    expectedTools: ["web_search"],
  },
  {
    task: "How many total spots are booked across Velvet Lounge and The Grove on Aug 1st combined?",
    expectedTools: ["query_color_db", "calculator"],
  },
  {
    task: "What's the mood tag for The Grove?",
    expectedTools: ["query_color_db"],
  },
  {
    task: "If a venue has 300 capacity and 210 booked, what percentage is that?",
    expectedTools: ["calculator"],
  },
  {
    task: "Search for the current price of Bitcoin and tell me what 0.5 BTC would be worth.",
    expectedTools: ["web_search", "calculator"],
  },
  {
    task: "Is Neon District closer to selling out than Velvet Lounge on Aug 1st?",
    expectedTools: ["query_color_db", "calculator"],
  },
  { task: "What's 47 times 89?", expectedTools: ["calculator"] },
  {
    task: "Look up today's date and tell me a fun fact about it.",
    expectedTools: ["web_search"],
  },
  {
    task: "What's the capacity of a venue called 'Skyline Room'?",
    expectedTools: ["query_color_db"],
  },
  {
    task: "Compare the moods of all three COLOR venues.",
    expectedTools: ["query_color_db"],
  },
  {
    task: "What's the square root concept — just explain it, no need to calculate anything specific.",
    expectedTools: [],
  },
  {
    task: "Search the web for the current weather in Sunnyvale, CA.",
    expectedTools: ["web_search"],
  },
  {
    task: "How much remaining capacity does Velvet Lounge have on Aug 2nd?",
    expectedTools: ["query_color_db", "calculator"],
  },
];

console.log("Starting eval...");
const runEval = async () => {
  console.log(`Running ${evalSet.length} Tasks...`);
  let correct = 0;
  const results = [];

  for (const item of evalSet) {
    const { toolsCalled } = await runAgent(item.task);
    const uniqueCalled = [...new Set(toolsCalled)];
    // Pass if every expected tool was called at least once (subset check,
    // not exact-sequence match -- per the "don't over-specify the path" advice)
    const passed =
      item.expectedTools.every((t) => uniqueCalled.includes(t)) &&
      (item.expectedTools.length > 0 || uniqueCalled.length === 0);

    if (passed) correct++;
    results.push({
      task: item.task,
      expected: item.expectedTools,
      actual: uniqueCalled,
      passed,
    });
    console.log(`${passed ? "✅" : "❌"} ${item.task.slice(0, 60)}`);
    console.log(
      `   expected: [${item.expectedTools.join(", ")}] | actual: [${uniqueCalled.join(", ")}]`,
    );
  }
  const accuracy = (correct / evalSet.length) * 100;
  console.log(
    `\nTool-selection accuracy: ${correct}/${evalSet.length} = ${accuracy.toFixed(1)}%`,
  );
};

runEval();
