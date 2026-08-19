import { runSolanaQuery } from "./solana_db.js";

const result = await runSolanaQuery({ limit: 3 });
console.log(result);
