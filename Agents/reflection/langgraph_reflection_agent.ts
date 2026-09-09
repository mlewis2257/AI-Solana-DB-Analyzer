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

const ReflectionState = Annotation.Root({
  ...MessagesAnnotation.spec,
  draftAnswer: Annotation<string>,
  critiqueFeedback: Annotation<string | null>,
  revisionCount: Annotation<number>({
    default: () => 0,
    reducer: (_, next) => next,
  }),
});

const plainModel = new ChatAnthropic({ model: "claude-sonnet-4-6" });

async function synthesizeNode(state: typeof ReflectionState.State) {
  try {
    const question = state.messages[0]?.content;
    const gatheredInfo = state.messages
      .map((m) => `${m.type}: ${m.content}`)
      .join("\n");

    if (!state.critiqueFeedback) {
      const response = await plainModel.invoke(
        `Question: ${question}\n\nInformation already gathered: ${gatheredInfo}\n\nGive the direct final answer.`,
      );
      return { draftAnswer: response.content };
    } else {
      const response = await plainModel.invoke(
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
  verdict: z.string(),
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

async function critiqueNode(state: typeof ReflectionState.State) {
  const question = state.messages[0]?.content;
  const answer = state.draftAnswer;
  const gatheredInfo = state.messages
    .map((m) => `${m.type}: ${m.content}`)
    .join("\n");
  try {
    const response = await critiqueModel.invoke(
      `You are reviewing an AI assistant's answer for correctness before it's shown to the user.

Question: ${question}

Answer: ${answer}

Raw data the assistant actually gathered (this IS the ground truth -- check the answer against THIS, not against what you'd expect to see):
${gatheredInfo}

Check for: logical/arithmetic errors, claims not supported by gathered information, answering a different question than asked, truncation, AND — if the answer presents a ranked or sorted list — verify the items are actually in correct order by the stated metric. A list that's numerically correct but out of order should be flagged as REVISE.`,
    );
    const toolCall = response.tool_calls?.[0];
    const verdict = toolCall?.args?.verdict ?? "APPROVED";
    const feedback = toolCall?.args?.feedback ?? "";

    const genuineRevision = verdict === "REVISE" && feedback.length > 0;
    return { critiqueFeedback: genuineRevision ? feedback : null };
  } catch (error) {
    return {
      critiqueFeedback: `An Error Critiquing Answer: ${(error as Error).message}`,
    };
  }
  // call the model with the submit_review tool forced, check state.draftAnswer
  // against the gathered data in state.messages, return { critiqueFeedback: ... }
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
