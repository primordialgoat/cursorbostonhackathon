import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-5"

LEAGUES = [
    "Brazilian Serie B",
    "Polish Ekstraklasa",
    "Norwegian Eliteserien",
    "Argentine Primera División",
    "Colombian Liga BetPlay",
    "Mexican Liga MX",
    "MLS",
]

DISCOVERY_LEAGUES = LEAGUES + ["Japanese J-League", "Korean K-League"]

KNOWN_PLAYERS = {
    "pedro rocha": {
        "age": 31, "club": "Coritiba", "position": "Left Winger",
        "goals": 15, "appearances": 32, "market_value": "€1.8M",
        "nationality": "Brazilian", "league": "Brazilian Serie B",
    },
    "efthymis koulouris": {
        "age": 28, "club": "Zagłębie Lubin", "position": "Striker",
        "goals": 28, "appearances": 34, "market_value": "€1.2M",
        "nationality": "Greek", "league": "Polish Ekstraklasa",
    },
    "karol czubak": {
        "age": 27, "club": "Radomiak", "position": "Striker",
        "goals": 16, "appearances": 27, "market_value": "€800K",
        "nationality": "Polish", "league": "Polish Ekstraklasa",
    },
    "lionel messi": {
        "age": 37, "club": "Inter Miami",
        "position": "Right Winger / Attacking Midfielder",
        "goals": 11, "assists": 8, "appearances": 19,
        "market_value": "€25M", "nationality": "Argentine", "league": "MLS",
    },
}

LEAGUE_QUALITY = {
    "Brazilian Serie B": (7, "Roughly equivalent to USL Championship level"),
    "Polish Ekstraklasa": (6, "Slightly below USL Championship"),
    "Norwegian Eliteserien": (5, "Comparable to USL League One"),
    "Argentine Primera División": (8, "Above USL Championship, approaching MLS level"),
    "Colombian Liga BetPlay": (6, "Slightly below USL Championship"),
    "Mexican Liga MX": (8, "Above USL Championship, approaching MLS level"),
    "Japanese J-League": (6, "Slightly below USL Championship"),
    "Korean K-League": (6, "Slightly below USL Championship"),
    "MLS": (9, "MLS — baseline comparison level"),
}

LEAGUE_ADJUSTMENT_FACTORS = {
    "Brazilian Serie B": {"goals": 0.65, "assists": 0.70, "label": "USL Championship equivalent"},
    "Polish Ekstraklasa": {"goals": 0.58, "assists": 0.62, "label": "Slightly below USL Championship"},
    "Norwegian Eliteserien": {"goals": 0.52, "assists": 0.55, "label": "USL League One equivalent"},
    "Argentine Primera División": {"goals": 0.72, "assists": 0.75, "label": "Above USL Championship"},
    "Mexican Liga MX": {"goals": 0.78, "assists": 0.80, "label": "Approaching MLS level"},
    "Colombian Liga BetPlay": {"goals": 0.55, "assists": 0.58, "label": "USL League One equivalent"},
    "MLS": {"goals": 1.0, "assists": 1.0, "label": "MLS — baseline comparison level"},
}

SYSTEM_PROMPT = """You are ScoutAI, an expert international soccer scout.
Produce a structured scouting report with these sections:

1. Player Overview
2. Technical Profile
3. Physical & Athletic Profile
4. Tactical Fit
5. Strengths
6. Weaknesses / Areas to Develop
7. Comparable Players
8. Projected Ceiling
9. Recommendation (Sign / Monitor / Pass) with reasoning

Be concise, specific, and avoid filler. Use markdown headings.

Research instructions:
If no specific stats are provided for the player, draw on your own knowledge of their recent career (current club, position, recent form, notable performances) to ground the report in real context. Do not fabricate exact numbers you do not know — qualify uncertain claims with hedges like "reportedly" or "around". If you have verified stats provided in the prompt, anchor on those.

League context:
Brazilian Serie B is roughly equivalent to USL Championship level. Polish Ekstraklasa is slightly below USL Championship. Norwegian Eliteserien is comparable to USL League One. Always include a specific sentence stating what the player's league translates to in American soccer terms."""


def format_known_player(data: dict) -> str:
    return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in data.items())


def build_user_prompt(player: str, league: str, stats: str) -> str:
    parts = [f"Player name: {player}", f"League: {league}"]
    if stats.strip():
        parts.append(f"Provided stats / notes:\n{stats.strip()}")
    parts.append("Write the scouting report now.")
    return "\n\n".join(parts)


def extract_verdict(report: str) -> Optional[str]:
    rec_match = re.search(r"Recommendation[^\n]*\n(.+)", report, flags=re.IGNORECASE | re.DOTALL)
    scope = rec_match.group(1) if rec_match else report
    cleaned = re.sub(r"\(\s*Sign\s*/\s*Monitor\s*/\s*Pass\s*\)", "", scope, flags=re.IGNORECASE)
    match = re.search(r"\b(SIGN|MONITOR|PASS)\b", cleaned, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_raw_stats(text: str) -> dict:
    if not text:
        return {}
    out = {}
    patterns = [
        ("goals", r"goals?\s*[:\-]?\s*(\d+)"),
        ("goals", r"(\d+)\s*goals?\b"),
        ("assists", r"assists?\s*[:\-]?\s*(\d+)"),
        ("assists", r"(\d+)\s*assists?\b"),
        ("appearances", r"appearances?\s*[:\-]?\s*(\d+)"),
        ("appearances", r"(\d+)\s*(?:appearances|apps|matches|games)\b"),
    ]
    for key, pattern in patterns:
        if key in out:
            continue
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                out[key] = int(match.group(1))
            except ValueError:
                pass
    return out


def compute_projection(raw: dict, league: str) -> Optional[dict]:
    factor = LEAGUE_ADJUSTMENT_FACTORS.get(league)
    if not factor or raw.get("goals") is None:
        return None
    raw_goals = raw["goals"]
    raw_assists = raw.get("assists", 0)
    apps = raw.get("appearances")
    adjusted_goals = raw_goals * factor["goals"]
    adjusted_assists = raw_assists * factor["assists"]
    proj = {
        "raw_goals": raw_goals,
        "raw_assists": raw_assists,
        "appearances": apps,
        "goals_factor": factor["goals"],
        "assists_factor": factor["assists"],
        "label": factor["label"],
        "adjusted_goals": round(adjusted_goals, 1),
        "adjusted_assists": round(adjusted_assists, 1),
    }
    if apps and apps > 0:
        proj["goals_per_90"] = round(adjusted_goals / apps, 2)
    return proj


def format_projection_for_prompt(proj: dict, league: str) -> str:
    apps_clause = f" in {proj['appearances']} apps" if proj.get("appearances") else ""
    gp90_clause = (
        f"\nProjected goals per 90 at USL Championship level: {proj['goals_per_90']}"
        if proj.get("goals_per_90") is not None else ""
    )
    return (
        f"[League-adjusted projection for {league}]\n"
        f"Raw season: {proj['raw_goals']} goals, {proj['raw_assists']} assists{apps_clause}\n"
        f"League adjustment factors: {proj['goals_factor']}x goals, {proj['assists_factor']}x assists ({proj['label']})\n"
        f"Projected USL Championship output: {proj['adjusted_goals']} goals, {proj['adjusted_assists']} assists"
        f"{gp90_clause}\n"
        f"Reference these adjusted numbers in your scouting report."
    )


_client_singleton: Optional[Anthropic] = None


def get_anthropic_client() -> Anthropic:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client_singleton


def generate_report(player: str, league: str, stats: str) -> str:
    client = get_anthropic_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(player, league, stats)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def discover_players(league: str, position: str, target_level: str) -> str:
    client = get_anthropic_client()
    prompt = (
        f"You are a soccer scout. List the top 5 players in {league} at position {position} "
        f"who would be realistic transfer targets for a {target_level} club. "
        f"For each player provide: name, age, club, key stats if known, USL translation rating out of 10, "
        f"and one sentence on why they fit. Format as a clean numbered list with markdown."
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def parse_player_names(markdown_text: str) -> list:
    names = []
    for raw_line in markdown_text.splitlines():
        line = re.sub(r"^[#>\-\*\s]+", "", raw_line)
        line = re.sub(r"^\*+", "", line)
        match = re.match(
            r"^\d+[\.\)]\s*\*{0,2}([^\n*\-:(,]+?)\*{0,2}(?:\s*[\-–—:(,]|\*\*|$)",
            line,
        )
        if match:
            name = match.group(1).strip().rstrip(",.")
            if name and 2 < len(name) <= 60:
                names.append(name)
    return names[:5]


def _extract_json_object(text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def enrich_player(player_name: str, league: str, target_level: str) -> Optional[dict]:
    client = get_anthropic_client()
    prompt = (
        f"Return only a JSON object for soccer player {player_name} in {league}. "
        f"Required fields: name (string), age (int), club (string), goals (int), assists (int), "
        f"appearances (int), market_value_euros (int, in euros), "
        f"usl_fit_score (int 1-10 measuring fit for a {target_level} club). "
        f"Use null for unknown numeric values. Output only the JSON object, no prose, no code fences."
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
    except Exception:
        return None
    data = _extract_json_object(text)
    if data is None:
        return None
    if not data.get("name"):
        data["name"] = player_name
    return data


def build_targets(league: str, position: str, target_level: str) -> list:
    listing = discover_players(league, position, target_level)
    names = parse_player_names(listing)
    if not names:
        return []
    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda n: enrich_player(n, league, target_level), names))
    return [r for r in results if r]


app = Flask(__name__, template_folder="templates")
CORS(app)


@app.route("/")
def index():
    return render_template(
        "report.html",
        leagues=LEAGUES,
        discovery_leagues=DISCOVERY_LEAGUES,
    )


@app.route("/api/scout", methods=["POST"])
def api_scout():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500
    data = request.get_json(silent=True) or {}
    player_name = (data.get("player_name") or "").strip()
    league = data.get("league") or ""
    stats = (data.get("stats") or "").strip()

    if not player_name:
        return jsonify({"error": "player_name is required"}), 400
    if league not in LEAGUE_QUALITY:
        return jsonify({"error": f"Unsupported league: {league}"}), 400

    known_key = player_name.lower()
    if known_key in KNOWN_PLAYERS:
        known = format_known_player(KNOWN_PLAYERS[known_key])
        combined_stats = (
            f"[Verified player data]\n{known}\n\n[Additional notes]\n{stats}"
            if stats else f"[Verified player data]\n{known}"
        )
    else:
        combined_stats = stats

    projection = compute_projection(parse_raw_stats(combined_stats), league)
    if projection:
        proj_block = format_projection_for_prompt(projection, league)
        combined_stats = f"{combined_stats}\n\n{proj_block}" if combined_stats else proj_block

    try:
        report = generate_report(player_name, league, combined_stats)
    except Exception as e:
        return jsonify({"error": f"Claude call failed: {e}"}), 502

    verdict = extract_verdict(report)
    score, label = LEAGUE_QUALITY[league]

    return jsonify({
        "verdict": verdict,
        "league_quality_score": score,
        "league_label": label,
        "projection": projection,
        "report_text": report,
    })


@app.route("/api/discover", methods=["POST"])
def api_discover():
    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500
    data = request.get_json(silent=True) or {}
    league = data.get("league") or ""
    position = data.get("position") or ""
    target_level = data.get("target_level") or ""

    if not league or not position or not target_level:
        return jsonify({"error": "league, position, and target_level are required"}), 400

    try:
        rows = build_targets(league, position, target_level)
    except Exception as e:
        return jsonify({"error": f"discover failed: {e}"}), 502

    return jsonify({"candidates": rows})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
