import {
  StateGraph,
  MemorySaver,
  MessagesAnnotation,
  Annotation,
  START,
} from "@langchain/langgraph";
import { toolsCondition, ToolNode } from "@langchain/langgraph/prebuilt";
import { tools, assistantNode } from "./langgraph_agent.js";
import { ChatAnthropic } from "@langchain/anthropic";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import type { BaseMessage } from "@langchain/core/messages";

export const ReflectionState = Annotation.Root({
  ...MessagesAnnotation.spec,
  recalledContext: Annotation<string>,
  draftAnswer: Annotation<string>,
  approved: Annotation<boolean>,
  critiqueFeedback: Annotation<string | null>,
  revisionCount: Annotation<number>({
    default: () => 0,
    reducer: (_, next) => next,
  }),
});

export const plainModel = new ChatAnthropic({ model: "claude-sonnet-4-6" });

export async function synthesizeNode(state: typeof ReflectionState.State) {
  try {
    const question = latestQuestion(state.messages);
    const gatheredInfo = state.messages
      .map((m) => `${m.type}: ${m.content}`)
      .join("\n");
    const memoryBlock = state.recalledContext
      ? `Context from earlier sessions:\n${state.recalledContext}\n\n` +
        `Follow the user preferences for format. Do not state a remembered ` +
        `insight as fact unless the gathered information supports it.\n\n`
      : "";
    if (!state.critiqueFeedback) {
      const response = await plainModel.invoke(
        memoryBlock +
          `Question: ${question}\n\nInformation already gathered: ${gatheredInfo}\n\nGive the direct final answer.`,
      );
      return { draftAnswer: response.content };
    } else {
      const response = await plainModel.invoke(
        memoryBlock +
          `Question: ${question}\n\nInformation already gathered: ${gatheredInfo}\n\n` +
          `Your Previous Answer Was: "${state.draftAnswer}"\n\n` +
          `A reviewer flagged this issue: ${state.critiqueFeedback}\n\n` +
          `Please provide a corrected answer.`,
      );
      return {
        draftAnswer: response.content,
        revisionCount: state.revisionCount + 1,
      };
    }
  } catch (error) {
    return {
      draftAnswer: `An Error Synthesizing Answer: ${(error as Error).message}`,
    };
  }
  // build a prompt from the gathered messages (+ critiqueFeedback if this is a revision),
  // call the model WITHOUT tools bound, return { draftAnswer: text }
}

const submitReviewSchema = z.object({
  verdict: z.enum(["APPROVED", "REVISE"]),
  feedback: z.string(),
});

type SubmitReviewInput = z.infer<typeof submitReviewSchema>;

const submitReviewTool = tool(
  async (input: SubmitReviewInput) => {
    return JSON.stringify(input);
  },
  {
    name: "submit_review",
    description:
      "Submit your verdict on whether the answer is correct and complete",
    schema: submitReviewSchema,
  },
);

const critiqueModel = new ChatAnthropic({
  model: "claude-sonnet-4-6",
}).bindTools([submitReviewTool], {
  tool_choice: "submit_review",
});

export async function critiqueNode(state: typeof ReflectionState.State) {
  const question = latestQuestion(state.messages);
  const answer = state.draftAnswer;
  const toolResults =
    state.messages
      .filter((m) => m.type === "tool")
      .map((m) => `${m.type}: ${m.content}`)
      .join("\n") || "(no tool calls were made)";
  try {
    const response = await critiqueModel.invoke(
      `You are reviewing an AI assistant's answer for correctness before it's shown to the user.

Question: ${question}

Answer: ${answer}

Raw data the assistant actually gathered (this IS the ground truth -- check the answer against THIS, not against what you'd expect to see):
${toolResults}

Check for: logical/arithmetic errors, claims not supported by gathered information, answering a different question than asked, truncation, AND — if the answer presents a ranked or sorted list — verify the items are actually in correct order by the stated metric. A list that's numerically correct but out of order should be flagged as REVISE.`,
    );
    const toolCall = response.tool_calls?.[0];
    const verdict = toolCall?.args?.verdict;
    const feedback = toolCall?.args?.feedback ?? "";
    console.log("--- CRITIQUE ---");
    if (!toolCall || !verdict) {
      console.error("critiqueNode: no verdict returned, answer is unreviewed");
      return { approved: false, critiqueFeedback: null };
    }
    if (verdict === "APPROVED") {
      console.log("critiqueNode: Verdict Approved, answer was reviewed");
      console.log("got a tool call:", Boolean(toolCall));
      console.log("verdict:", verdict);
      return { approved: true, critiqueFeedback: null };
    }

    if (verdict === "REVISE" && feedback) {
      console.log("critiqueNode: Verdict is revised");
      console.log("got a tool call:", Boolean(toolCall));
      console.log("feedback:", feedback || "(none)");
      return { approved: false, critiqueFeedback: feedback };
    }

    return { approved: false, critiqueFeedback: null };
  } catch (error) {
    console.error("Error critiquing answer", error);
    return {
      critiqueFeedback: null,
      approved: false,
    };
  }
  // call the model with the submit_review tool forced, check state.draftAnswer
  // against the gathered data in state.messages, return { critiqueFeedback: ... }
}

export function latestQuestion(messages: BaseMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.type === "human") {
      const content = messages[i]?.content;
      if (typeof content === "string") {
        return content;
      }
      if (Array.isArray(content)) {
        return content
          .map((part) => {
            if (typeof part === "string") {
              return part;
            }
            if ("text" in part && typeof part.text === "string") {
              return part.text;
            }
            return JSON.stringify(part);
          })
          .join("");
      }
      return "";
    }
  }
  return "";
}

function routeAfterCritique(state: typeof ReflectionState.State) {
  if (state.critiqueFeedback && state.revisionCount < 2) {
    return "synthesize";
  }
  return "__end__";
}

const builder = new StateGraph(ReflectionState)
  .addNode("assistant", assistantNode)
  .addNode("tools", new ToolNode(tools))
  .addNode("synthesize", synthesizeNode)
  .addNode("critique", critiqueNode)
  .addEdge("__start__", "assistant")
  .addConditionalEdges("assistant", toolsCondition, {
    tools: "tools",
    __end__: "synthesize",
  })
  .addEdge("tools", "assistant")
  .addEdge("synthesize", "critique")
  .addConditionalEdges("critique", routeAfterCritique);

const checkpointer = new MemorySaver();
export const reflectionGraph = builder.compile({ checkpointer });

// async function main() {
//   const result = await refelctionGraph.invoke(
//     {
//       messages: [
//         {
//           role: "user",
//           content: "What is 847 times 23, plus 100?",
//         },
//       ],
//     },
//     { configurable: { thread_id: "reflection_thread_1" } },
//   );

//   console.log("Final Revision:", result.draftAnswer);
//   console.log("Revisions Needed:", result.revisionCount);
// }

async function main() {
  const result = await reflectionGraph.invoke(
    {
      messages: [
        {
          role: "user",
          content:
            "What are the top 10 highest PnL trades from both Strategy A and B, ranked together correctly?",
        },
      ],
    },
    {
      configurable: { thread_id: "reflection_test_2" },
    },
  );
  console.log("Final Revision:", result.draftAnswer);
  console.log("Number of Revisions:", result.revisionCount);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
