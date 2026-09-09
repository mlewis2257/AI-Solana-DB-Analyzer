// test_langsmith_direct.ts
import dotenv from "dotenv";
dotenv.config();

fetch("https://api.smith.langchain.com/info", {
  headers: { "x-api-key": process.env.LANGSMITH_API_KEY! },
})
  .then((r) => r.json())
  .then(console.log)
  .catch((e) => console.error("REAL ERROR:", e.cause || e));
