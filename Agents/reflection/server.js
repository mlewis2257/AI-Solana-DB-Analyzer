import express from "express";
import { WebSocketServer } from "ws";
import { runAgent } from "./agent.js";
import { runAgentWithReflection } from "./reflection.js";

const app = express();
app.use(express.static("public"));

const server = app.listen(3000, () => {
  console.log("listening on http://localhost:3000");
});

const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  ws.on("message", async (raw) => {
    const { query } = JSON.parse(raw.toString());
    const timeout = setTimeout(() => {
      ws.send(
        JSON.stringify({
          type: "error",
          message: "Request timed out after 60s",
        }),
      );
    }, 60000);
    try {
      await runAgentWithReflection(query, (event) => {
        ws.send(JSON.stringify(event));
      });
    } catch (err) {
      ws.send(JSON.stringify({ type: "error", message: err.message }));
    } finally {
      clearTimeout(timeout);
    }
  });
});
