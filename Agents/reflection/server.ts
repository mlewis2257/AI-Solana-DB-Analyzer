import express from "express";
import { WebSocketServer } from "ws";
// import { runAgent } from "./agent.js";
import { runAgentWithReflection } from "./reflection.js";
import { reflectionGraph } from "./langgraph_reflection_agent.js";
import { AIMessage, ToolMessage } from "@langchain/core/messages";

const app = express();
app.use(express.static("public"));

const server = app.listen(3000, () => {
  console.log("listening on http://localhost:3000");
});

const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  ws.on("message", async (raw) => {
    const { query } = JSON.parse(raw.toString());
    const config = { configurable: { thread_id: `req-${Date.now()}` } };
    let seenGathering = false;

    try {
      const stream = await reflectionGraph.stream(
        {
          messages: [{ role: "user", content: query }],
        },
        { ...config, streamMode: "updates" },
      );

      for await (const chunk of stream) {
        for (const [nodeName, update] of Object.entries(chunk) as [
          string,
          any,
        ][]) {
          if (nodeName === "assistant") {
            if (!seenGathering) {
              ws.send(JSON.stringify({ type: "phase", phase: "gathering" }));
              seenGathering = true;
            }
            const aiMessage = update.message[0] as AIMessage;
            for (const tc of aiMessage.tool_calls ?? []) {
              ws.send(
                JSON.stringify({
                  type: "tool_call",
                  tool: tc.name,
                  input: tc.args,
                }),
              );
            }
          } else if (nodeName === "tools") {
            for (const tm of update.messages as ToolMessage[]) {
              ws.send(
                JSON.stringify({
                  type: "tool_result",
                  tool: tm.name,
                  result: tm.content,
                }),
              );
            }
          } else if (nodeName === "synthesize") {
            ws.send(JSON.stringify({ type: "phase", phase: "synthesize" }));
            ws.send(
              JSON.stringify({
                type: "synthesize_result",
                text: update.draftAnswer,
              }),
            );
          } else if (nodeName === "critique") {
            ws.send(JSON.stringify({ type: "phase", phase: "critique" }));
            ws.send(
              JSON.stringify({
                type: "critique_result",
                verdict: update.critiqueFeedback ? "REVISE" : "APPROVED",
                feedback: update.critiqueFeedback,
              }),
            );
          }
        }
      }
      const finalState = await reflectionGraph.getState(config);
      ws.send(
        JSON.stringify({
          type: "done",
          finalText: finalState.values.draftAnswer,
          genuineRevision: finalState.values.revisionCount > 0,
          toolsCalled: [],
        }),
      );
    } catch (error) {
      ws.send(
        JSON.stringify({ type: "error", message: (error as Error).message }),
      );
    }
    // const timeout = setTimeout(() => {
    //   ws.send(
    //     JSON.stringify({
    //       type: "error",
    //       message: "Request timed out after 60s",
    //     }),
    //   );
    // }, 60000);
    // try {
    //   await runAgentWithReflection(query, (event) => {
    //     ws.send(JSON.stringify(event));
    //   });
    // } catch (err) {
    //   ws.send(JSON.stringify({ type: "error", message: err.message }));
    // } finally {
    //   clearTimeout(timeout);
    // }
  });
});
