import { runAgent } from "./agent.js";
import { Anthropic } from "@anthropic-ai/sdk/client.js";
import { config } from "dotenv";
// import { on } from "ws";

const anthropic = new Anthropic(process.env);

const synthesize = async (
  question,
  gatheredInfo,
  onEvent = () => {},
  max_tokens = 1500,
) => {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: max_tokens,
    system: `Answer directly and concisely. If the question involves a term with multiple common definitions or measurements (e.g. a city's population meaning different things depending on boundaries used), state which specific definition your number refers to in a few words — don't hide the ambiguity, but don't over-explain it either. One to two sentences max.`,
    messages: [
      {
        role: "user",
        content: `Question: ${question}\n\nInformation already gathered: ${gatheredInfo}\n\nGive the direct final answer.`,
      },
    ],
  });
  const text = response.content.find((b) => b.type === "text")?.text ?? "";
  onEvent({ type: "Synthesis Result", text });
  return text;
};

export const critiqueAgentResponse = async (
  question,
  answer,
  groundingData,
  onEvent = () => {},
) => {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 300,
    tools: [
      {
        name: "submit_review",
        description:
          "Submit your verdict on whether the answer is correct and complete",
        input_schema: {
          type: "object",
          properties: {
            verdict: { type: "string", enum: ["APPROVED", "REVISE"] },
            feedback: {
              type: "string",
              description:
                "Required and must be non-empty if verdict is REVISE — explain specifically what's wrong. Empty string only if verdict is APPROVED.",
            },
          },
          required: ["verdict", "feedback"],
        },
      },
    ],
    tool_choice: { type: "tool", name: "submit_review" },
    messages: [
      {
        role: "user",
        content: `You are reviewing an AI assistant's answer for correctness before it's shown to the user.

Question: ${question}
Answer: ${answer}

Raw data the assistant actually gathered (this IS the ground truth -- check the answer against THIS, not against what you'd expect to see):
${groundingData}

Check for: logical/arithmetic errors, claims not supported by gathered information, answering a different question than asked, truncation, AND — if the answer presents a ranked or sorted list — verify the items are actually in correct order by the stated metric. A list that's numerically correct but out of order should be flagged as REVISE.`,
      },
    ],
  });
  const toolUse = response.content.find((b) => b.type === "tool_use");
  //   console.log("RAW critique tool call:", JSON.stringify(toolUse?.input));
  const result = toolUse
    ? toolUse.input
    : { verdict: "APPROVED", feedback: "" };
  const verdict = result.verdict ?? "APPROVED";
  const feedback =
    result.feedback || "(model flagged REVISE but did not explain why)";
  const genuineRevision =
    verdict === "REVISE" && result.feedback && result.feedback.length > 0;
  onEvent({
    type: "critique_result",
    verdict,
    feedback: genuineRevision ? feedback : null,
  });

  return { verdict, feedback, genuineRevision };
};

export const runAgentWithReflection = async (userQuery, onEvent = () => {}) => {
  console.log(">>> Starting gather phase");
  onEvent({ type: "phase", phase: "gathering" });
  const gathered = await runAgent(userQuery);
  console.log(
    ">>> Gather phase complete, starting synthesis",
    gathered.toolsCalled,
  );

  console.log("Synthesizing Phase Started");
  let finalText = await synthesize(userQuery, gathered.finalText);
  console.log(">>> Synthesize phase complete:", finalText.slice(0, 80));

  console.log(">>> Starting critique phase");
  let { verdict, feedback, genuineRevision } = await critiqueAgentResponse(
    userQuery,
    finalText,
    gathered.finalText,
  );
  console.log(">>> Critique phase complete");

  const looksTruncated =
    genuineRevision && /truncat|cut off|incomplete/i.test(feedback);
  if (looksTruncated) {
    console.log(">>> Detected truncation, retrying synthesis with more room");
    onEvent({ type: "phase", phase: "retrying_synthesis" });
    finalText = await synthesize(userQuery, gathered.finalText, onEvent, 2500);
    console.log(">>> Retry Synthesizing Phase:", finalText.slice(0, 80));

    const recheck = await critiqueAgentResponse(
      userQuery,
      finalText,
      gathered.finalText,
      onEvent,
    );
    verdict = recheck.verdict;
    feedback = recheck.feedback;
    genuineRevision = recheck.genuineRevision;
    console.log(">>> Recheck Complete");
  }

  const result = {
    finalText,
    verdict,
    genuineRevision,
    critiqueNote: genuineRevision ? feedback : null,
    toolsCalled: gathered.toolsCalled,
    wasRetried: looksTruncated,
  };
  onEvent({ type: "done", ...result });
  console.log(">>> Done event sent");
  return result;
};
