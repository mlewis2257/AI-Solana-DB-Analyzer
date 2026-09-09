# Solana Trade Analyst — Reflective Tool-Calling Agent

A production-data-grounded AI agent that answers questions about a live Solana trading bot's historical performance — built twice (raw Anthropic SDK, then ported to LangGraph/TypeScript), with a self-critique loop that catches its own reasoning mistakes before they reach the user.

## What this actually does

Ask it a real question about trading history — *"Compare Strategy A and Strategy B's average PnL on closed trades,"* *"What are the top 10 highest PnL trades across both strategies?"*, *"Is there a VIP tier that outperforms untiered trades?"* — and it:

1. **Gathers** real data using tools (a read-only query against a live Postgres database, web search, live crypto price lookups, or a calculator)
2. **Synthesizes** a direct answer from what it found — using a separate, tools-unbound model instance so it can't silently re-fetch data instead of reasoning over what it already has
3. **Critiques** its own answer against the raw gathered data, checking for arithmetic errors, unsupported claims, incorrect sort order, and truncation
4. **Revises** automatically if the critique finds a real problem, up to a bounded retry limit — then returns the final, checked answer

## Why the reflection loop matters (not just a nice-to-have)

The strongest result in this project's development wasn't a bug fix — it was a case where the base model **fabricated a complete, confident-looking statistics table** in response to an open-ended question ("what are the best trading days of the week?"), and the critique caught it across three consecutive revision attempts before the system settled on an honest answer: *the data doesn't support a confident conclusion here.* See `week4-phase2-solana-findings.md` for the full writeup, including the actual critique feedback that caught the fabrication.

## Architecture

```
User query
  → assistant (Claude + tools) ⟲ tools node   [gathers real data, loops until done]
  → synthesize (tools-unbound model)           [drafts an answer from gathered data]
  → critique (forced structured-output tool)   [checks the draft against raw data]
      ├─ REVISE (with real feedback) → back to synthesize
      └─ APPROVED, or revision cap hit → done
```

Two parallel implementations exist in this repo:
- **`agent.js` / `reflection.js`** — hand-built using the raw `@anthropic-ai/sdk`, no framework. Built first, to understand every mechanic directly (the tool-call loop, container-based server tools, message-state threading) before adopting a framework.
- **`langgraph_agent.ts` / `langgraph_reflection_agent.ts`** — the same architecture rebuilt in LangGraph, using `StateGraph`, `ToolNode`, `toolsCondition`, and `MemorySaver`, with TypeScript and `zod` schemas throughout.

## Tools

| Tool | Purpose |
|---|---|
| `calculator` | Basic arithmetic, sandboxed via `Function()` |
| `query_solana_trades` | Parameterized, read-only query against the trading bot's Postgres database (`trading_positions`, `tokens`, `calls`). Supports filtering, aggregation (`COUNT`/`AVG`/`SUM`/`MIN`/`MAX`), and sortable/directional ranking |
| `tavily_search` (LangGraph version) / `web_search` (raw SDK version) | General web search for anything not in the trading database |
| `get_crypto_price` | Live price lookups via CoinGecko — added after discovering that web search alone returns stale, article-derived prices unsuitable for "what's the current price" questions |

## Security posture

- Database access uses a dedicated `agent_readonly` Postgres role — `SELECT`-only, enforced at the database level, not just in application code
- All SQL queries are parameterized (`$1`, `$2`, ...) — no string-concatenated values, since tool arguments ultimately originate from LLM-decided input and are treated as untrusted
- Database connectivity runs through an SSH tunnel rather than an open firewall port
- Secrets (`.env`) are excluded from the Docker build context (`.dockerignore`) and injected only at container-start time via `docker-compose.yml`'s `env_file` — never baked into the image itself

## Evaluation

A 15-question golden set covering all four tools plus two no-tool-needed control questions, scored on tool-selection accuracy using a subset-check methodology (every *expected* tool must be called; exact call sequence isn't over-specified). Current result: **15/15**, after correcting one stale eval label left over from the pre-LangGraph-port tool naming.

## Running it

**Prerequisites:** Node 20+, an Anthropic API key, a Tavily API key (free tier), access to the Solana trading bot's Postgres database via SSH tunnel.

```bash
npm install
```

**Open the SSH tunnel** (in its own terminal, keep it running):
```bash
npm run tunnel
```

**Set up `.env`:**
```
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
SOLANA_DB_URL=postgresql://agent_readonly:password@localhost:5433/solana_signals
```

**Run directly:**
```bash
npx tsx langgraph_reflection_agent.ts
```

**Or via Docker** (update `SOLANA_DB_URL`'s host to `host.docker.internal` first, since `localhost` inside a container refers to the container itself):
```bash
docker compose up
```

Then visit `http://localhost:3000` for the live WebSocket frontend, showing every phase (gathering, synthesizing, critiquing) as it streams.

## Known limitations

- **LangSmith tracing is not currently wired up** — attempted, hit a confirmed, documented upstream bug in the JS SDK's multipart trace-ingestion path (isolated through systematic elimination: network, API key, and SDK version were all confirmed fine). Currently relying on manual console logging instead.
- **No MCP server yet** — tools are currently bound directly into this agent's own process; not yet exposed as a standalone MCP server other clients could connect to.
- **No cross-session long-term memory** — `MemorySaver` persists within a single thread, but there's no `Store`-backed memory letting the agent recall context across separate conversations yet.
- Sort direction (`ORDER BY ... ASC/DESC`) and VIP-tier "untiered" filtering were both real bugs found and fixed during development — see the findings doc for the full list of debugged issues.

## Roadmap

1. MCP server wrapping `query_solana_trades` (and possibly `calculator`), so any MCP-compatible client can query this data directly
2. A GTM-style prospecting workflow (discovery → research → ICP scoring → outreach drafting → fabrication-check), reusing the same reflection pattern for a different domain
3. Cross-session memory via LangGraph's `Store`
4. A larger, more adversarial evaluation set
