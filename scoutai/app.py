import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL = "claude-sonnet-4-5"

LEAGUES = [
    "Brazilian Serie B",
    "Polish Ekstraklasa",
    "Norwegian Eliteserien",
    "Argentine Primera División",
    "Colombian Liga BetPlay",
    "Japanese J-League",
    "Korean K-League",
    "Mexican Liga MX",
]

LEAGUE_QUALITY = {
    "Brazilian Serie B": (7, "Roughly equivalent to USL Championship level"),
    "Polish Ekstraklasa": (6, "Slightly below USL Championship"),
    "Norwegian Eliteserien": (5, "Comparable to USL League One"),
    "Argentine Primera División": (8, "Above USL Championship, approaching MLS level"),
    "Colombian Liga BetPlay": (6, "Slightly below USL Championship"),
    "Japanese J-League": (6, "Slightly below USL Championship"),
    "Korean K-League": (6, "Slightly below USL Championship"),
    "Mexican Liga MX": (8, "Above USL Championship, approaching MLS level"),
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

League context:
Brazilian Serie B is roughly equivalent to USL Championship level. Polish Ekstraklasa is slightly below USL Championship. Norwegian Eliteserien is comparable to USL League One. Always include a specific sentence stating what the player's league translates to in American soccer terms."""


def build_user_prompt(player: str, league: str, stats: str) -> str:
    parts = [
        f"Player name: {player}",
        f"League: {league}",
    ]
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


STAT_FIELDS = ["goals", "assists", "appearances", "position", "age", "club"]


def fetch_player_stats(player_name: str, league: str) -> Optional[str]:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = (
        f"Return only a JSON object with real stats for {player_name} in {league}. "
        f"Fields: goals, assists, appearances, position, age, club. "
        f"If you don't know exact stats, use null for that field."
    )
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group(0))
        lines = []
        for key in STAT_FIELDS:
            value = data.get(key)
            if value is not None:
                lines.append(f"{key.capitalize()}: {value}")
        return "\n".join(lines) if lines else None
    except Exception:
        return None


def generate_report(player: str, league: str, stats: str) -> str:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(player, league, stats)}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


st.set_page_config(page_title="ScoutAI", page_icon=None, layout="centered")

st.title("ScoutAI")
st.caption("International soccer scouting, powered by Claude.")

player_name = st.text_input("Player name")
league = st.selectbox("League", LEAGUES)
stats = st.text_area("Stats (optional)", height=150, placeholder="Goals, assists, xG, minutes, etc.")

if st.button("Generate scouting report", type="primary"):
    if not ANTHROPIC_API_KEY:
        st.error("ANTHROPIC_API_KEY is not set. Add it at the top of app.py.")
    elif not player_name.strip():
        st.warning("Enter a player name.")
    else:
        with st.spinner("Scouting..."):
            try:
                live_stats = fetch_player_stats(player_name, league)
                if live_stats:
                    st.caption("📚 Stats sourced from Claude knowledge base")
                    combined_stats = (
                        f"[Claude-sourced stats]\n{live_stats}\n\n[Additional notes]\n{stats.strip()}"
                        if stats.strip()
                        else f"[Claude-sourced stats]\n{live_stats}"
                    )
                else:
                    combined_stats = stats
                report = generate_report(player_name, league, combined_stats)
                verdict = extract_verdict(report)
                if verdict == "SIGN":
                    st.success("### Verdict: SIGN ✅")
                elif verdict == "MONITOR":
                    st.warning("### Verdict: MONITOR ⚠️")
                elif verdict == "PASS":
                    st.error("### Verdict: PASS ❌")
                score, label = LEAGUE_QUALITY[league]
                st.subheader("League Quality vs USL")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label=league, value=f"{score}/10")
                with col2:
                    st.progress(score / 10)
                    st.caption(label)
                st.divider()
                st.markdown(report)
            except Exception as e:
                st.error(f"Request failed: {e}")
