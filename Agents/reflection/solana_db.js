import pg from "pg";
import dotenv from "dotenv";

dotenv.config();

const pool = new pg.Pool({ connectionString: process.env.SOLANA_DB_URL });

export const runSolanaQuery = async (input) => {
  const {
    symbol,
    status,
    is_strategy_b,
    exit_reason,
    min_pnl_pct,
    max_pnl_pct,
    min_conviction_score,
    max_conviction_score,
    vip_tier,
    order_by = "exit_time",
    order_direction = "desc",
    aggregate = false,
    limit = 20,
  } = input;

  const conditions = [];
  const values = [];
  let i = 1;

  if (symbol !== undefined) {
    conditions.push(`t.symbol ILIKE $${i++}`);
    values.push(symbol);
  }
  if (status !== undefined) {
    conditions.push(`p.status = $${i++}`);
    values.push(status);
  }
  if (is_strategy_b !== undefined) {
    conditions.push(`p.is_strategy_b = $${i++}`);
    values.push(is_strategy_b);
  }
  if (exit_reason !== undefined) {
    conditions.push(`p.exit_reason = $${i++}`);
    values.push(exit_reason);
  }
  if (min_pnl_pct !== undefined) {
    conditions.push(`p.pnl_pct >= $${i++}`);
    values.push(min_pnl_pct);
  }
  if (max_pnl_pct !== undefined) {
    conditions.push(`p.pnl_pct <= $${i++}`);
    values.push(max_pnl_pct);
  }
  if (min_conviction_score !== undefined) {
    conditions.push(`c.conviction_score >= $${i++}`);
    values.push(min_conviction_score);
  }
  if (max_conviction_score !== undefined) {
    conditions.push(`c.conviction_score <= $${i++}`);
    values.push(max_conviction_score);
  }
  if (vip_tier !== undefined) {
    if (vip_tier === "none" || vip_tier === "untiered") {
      conditions.push(`p.vip_tier IS NULL`);
    } else {
      conditions.push(`p.vip_tier = $${i++}`);
      values.push(vip_tier);
    }
  }

  const where =
    conditions.length > 0 ? `WHERE ${conditions.join(" AND ")}` : "";

  const ALLOWED_SORTED_COLUMNS = {
    exit_time: "p.exit_time",
    entry_time: "p.entry_time",
    pnl_pct: "p.pnl_pct",
    peak_multiplier: "p.peak_multiplier",
    conviction_score: "c.conviction_score",
  };

  const sortColumn = ALLOWED_SORTED_COLUMNS[order_by] || "p.exit_time";
  const direction = order_direction === "asc" ? "ASC" : "DESC";

  const safeLimit = Math.min(Math.max(parseInt(limit) || 20, 1), 100);

  const query = aggregate
    ? `     SELECT
        COUNT(*) AS trade_count,
        AVG(p.pnl_pct) AS avg_pnl_pct,
        SUM(p.pnl_sol) AS total_pnl_sol,
        MIN(p.pnl_pct) AS min_pnl_pct,
        MAX(p.pnl_pct) AS max_pnl_pct
        FROM trading_positions p
        LEFT JOIN tokens t ON t.id = p.token_id
        LEFT JOIN calls c ON c.id = p.call_id
        ${where}
`
    : `
           SELECT
            p.id, t.symbol, t.name, p.status,
            p.entry_price, p.entry_mcap, p.entry_time,
            p.exit_price, p.exit_mcap, p.exit_time,
            p.pnl_sol, p.pnl_pct, p.exit_reason,
            p.is_strategy_b, p.vip_tier, p.peak_mcap, p.peak_multiplier,
            p.is_simulation,
            c.conviction_score, c.narrative_tags, c.source_platform
            FROM trading_positions p
            LEFT JOIN tokens t ON t.id = p.token_id
            LEFT JOIN calls c ON c.id = p.call_id
            ${where}
            ORDER BY ${sortColumn} ${direction} DESC NULLS LAST
            LIMIT ${safeLimit}
           `;
  console.log("--- QUERY ---");
  console.log(query);
  console.log("--- VALUES ---", values);

  try {
    const result = await pool.query(query, values);
    return JSON.stringify(result.rows);
  } catch (error) {
    return `Database error: ${error.message}`;
  }
};
