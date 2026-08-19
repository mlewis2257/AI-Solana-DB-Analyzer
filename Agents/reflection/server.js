import express from "express";
import { WebSocketServer } from "ws";
import { runAgent } from "./agent.js";

const app = express();
app.use(express.static("public"));

const server = app.listen(3000, () => {
  console.log("listening on http://localhost:3000");
});

const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  ws.on("message", async (raw) => {
    const { query } = JSON.parse(raw.toString());
    try {
      await runAgent(query, (event) => {
        ws.send(JSON.stringify(event));
      });
    } catch (error) {
      ws.send(JSON.stringify({ type: "error", message: error.message }));
    }
  });
});
