import os

import streamlit as st
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL = "claude-sonnet-4-5"

LEAGUES = [
    "Brazilian Serie B",
    "Polish Ekstraklasa",
    "Norwegian Eliteserien",
]

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
                report = generate_report(player_name, league, stats)
                st.markdown(report)
            except Exception as e:
                st.error(f"Request failed: {e}")
