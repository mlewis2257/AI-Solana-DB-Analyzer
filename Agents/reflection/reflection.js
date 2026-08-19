import { runAgent } from "./agent.js";
import { Anthropic } from "@anthropic-ai/sdk/client.js";
import { config } from "dotenv";

const anthropic = new Anthropic(process.env);

const synthesize = async (question, gatheredInfo) => {
  const response = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 512,
    system: `Answer directly and concisely. If the question involves a term with multiple common definitions or measurements (e.g. a city's population meaning different things depending on boundaries used), state which specific definition your number refers to in a few words — don't hide the ambiguity, but don't over-explain it either. One to two sentences max.`,
    messages: [
      {
        role: "user",
        content: `Question: ${question}\n\nInformation already gathered: ${gatheredInfo}\n\nGive the direct final answer.`,
      },
    ],
  });
  return response.content.find((b) => b.type === "text")?.text ?? "";
};

export const critiqueAgentResponse = async (
  question,
  answer,
  groundingData,
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

Check for: logical/arithmetic errors, claims not supported by gathered information, answering a different question than asked, truncation.`,
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
  return { verdict, feedback, genuineRevision };
};

export const runAgentWithReflection = async (userQuery) => {
  const gathered = await runAgent(userQuery);
  const finalText = await synthesize(userQuery, gathered.finalText);
  const { verdict, feedback, genuineRevision } = await critiqueAgentResponse(
    userQuery,
    finalText,
    gathered.finalText,
  );

  return {
    finalText,
    verdict,
    genuineRevision,
    critiqueNote: genuineRevision ? feedback : null,
    toolsCalled: gathered.toolsCalled,
  };
};
