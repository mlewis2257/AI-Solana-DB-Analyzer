import asyncio
import os
from dotenv import load_dotenv
import requests
from typing import Any
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, tool

load_dotenv()

BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY")
BASE_URL = "https://api.balldontlie.io/nba/v1"
HEADERS = {"Authorization": BALLDONTLIE_API_KEY}


@tool(
    "get_todays_games",
    "Gets today's NBA games, including scores if in progress or final, and status (scheduled/live/final)",
    {}
)
async def get_todays_games(args: dict[str, Any]) -> dict[str, Any]:
    from datetime import date
    today = date.today().isoformat()
    resp = requests.get(f"{BASE_URL}/games",
                        headers=HEADERS, params={"dates[]": today})
    if resp.status_code != 200:
        return {"content": [{"type": "text", "text": f"API error {resp.status_code}: {resp.text[:200]}"}]}
    games = resp.json().get("data", [])
    if not games:
        return {"content": [{"type": "text", "text": "No NBA games scheduled today."}]}
    lines = []
    for g in games:
        home, away = g["home_team"]["full_name"], g["visitor_team"]["full_name"]
        lines.append(
            f"{away} {g['visitor_team_score']} @ {home} {g['home_team_score']} — {g['status']}")
    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


@tool(
    "get_team_record",
    "Gets a team's win-loss record this season by counting their completed games",
    {"team_name": str}

)
async def get_team_record(args: dict[str, Any]) -> dict[str, Any]:
    team_name = args["team_name"]

    # First, find the team's ID by searching teams (free tier)
    teams_resp = requests.get(f"{BASE_URL}/teams", headers=HEADERS)
    if teams_resp.status_code != 200:
        return {"content": [{"type": "text", "text": f"API error {teams_resp.status_code}: {teams_resp.text[:200]}"}]}
    teams = teams_resp.json().get("data", [])
    team = next((t for t in teams if team_name.lower()
                in t["full_name"].lower()), None)
    if not team:
        return {"content": [{"type": "text", "text": f"No team found matching '{team_name}'."}]}

    # Pull this team's completed games this season (free tier)
    games_resp = requests.get(
        f"{BASE_URL}/games",
        headers=HEADERS,
        params={"team_ids[]": team["id"], "seasons[]": 2025, "per_page": 100},
    )
    if games_resp.status_code != 200:
        return {"content": [{"type": "text", "text": f"API error {games_resp.status_code}: {games_resp.text[:200]}"}]}
    games = games_resp.json().get("data", [])

    wins, losses = 0, 0
    for g in games:
        if g["status"] != "Final":
            continue
        is_home = g["home_team"]["id"] == team["id"]
        team_score = g["home_team_score"] if is_home else g["visitor_team_score"]
        opp_score = g["visitor_team_score"] if is_home else g["home_team_score"]
        if team_score > opp_score:
            wins += 1
        else:
            losses += 1

    text = f"{team['full_name']}: {wins}-{losses} this season ({wins + losses} games completed)"
    return {"content": [{"type": "text", "text": text}]}


async def main() -> None:
    sports_server = create_sdk_mcp_server(
        name="sports",
        version="1.0.0",
        tools=[get_todays_games, get_team_record]


    )

    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        system_prompt="You are a knowledgeable NBA analyst who answers questions using live game and standings data.",
        mcp_servers={"sports": sports_server},
        allowed_tools=["mcp__sports__get_todays_games",
                       "mcp__sports__get_team_record"]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("What NBA games are on today, and how are the San Antonio Spurs doing this season?")
        await client.query("What NBA games are on today, and how are the New York Knicks doing this season?")
        await client.query("Who won the NBA Championship in 2026?")
        await client.query("How good were the Chicago Bulls in 1996?")

        async for message in client.receive_messages():
            event = filter_message(message)
            print(event)


def filter_message(message) -> dict | None:
    """Turn a raw SDK message into a clean, frontend-ready event, or None to skip it."""
    msg_type = type(message).__name__

    if msg_type == "AssistantMessage":
        for block in message.content:
            block_type = type(block).__name__
            if block_type == "TextBlock":
                return {"type": "thinking", "text": block.text}
            elif block_type == "ToolUseBlock":
                return {"type": "tool_call", "tool": block.name, "input": block.input}
            # ThinkingBlock (the raw reasoning trace) intentionally skipped --
            # too granular/internal for a user-facing stream

    elif msg_type == "UserMessage":
        for block in message.content:
            if type(block).__name__ == "ToolResultBlock":
                # content is usually a list of {"type": "text", "text": ...} dicts
                text = block.content
                if isinstance(text, list) and text and "text" in text[0]:
                    text = text[0]["text"]
                return {"type": "tool_result", "result": text, "is_error": bool(block.is_error)}

    elif msg_type == "ResultMessage":
        return {"type": "final_answer", "text": message.result, "cost_usd": message.total_cost_usd}

    # SystemMessage (init, thinking_tokens), RateLimitEvent -- all skipped, internal-only
    return None


if __name__ == "__main__":
    asyncio.run(main())
