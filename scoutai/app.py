import json
import os
import re
import smtplib
from concurrent.futures import ThreadPoolExecutor
from email.message import EmailMessage
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st
from anthropic import Anthropic
from fpdf import FPDF

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MODEL = "claude-sonnet-4-5"

LEAGUES = [
    "Brazilian Serie B",
    "Polish Ekstraklasa",
    "Norwegian Eliteserien",
    "Argentine Primera División",
    "Colombian Liga BetPlay",
    "Mexican Liga MX",
]

DISCOVERY_LEAGUES = LEAGUES + ["Japanese J-League", "Korean K-League"]

KNOWN_PLAYERS = {
    "pedro rocha": {
        "age": 31,
        "club": "Coritiba",
        "position": "Left Winger",
        "goals": 15,
        "appearances": 32,
        "market_value": "€1.8M",
        "nationality": "Brazilian",
        "league": "Brazilian Serie B",
    },
    "efthymis koulouris": {
        "age": 28,
        "club": "Zagłębie Lubin",
        "position": "Striker",
        "goals": 28,
        "appearances": 34,
        "market_value": "€1.2M",
        "nationality": "Greek",
        "league": "Polish Ekstraklasa",
    },
    "karol czubak": {
        "age": 27,
        "club": "Radomiak",
        "position": "Striker",
        "goals": 16,
        "appearances": 27,
        "market_value": "€800K",
        "nationality": "Polish",
        "league": "Polish Ekstraklasa",
    },
}


def format_known_player(data: dict) -> str:
    return "\n".join(f"{key.replace('_', ' ').capitalize()}: {value}" for key, value in data.items())


LEAGUE_QUALITY = {
    "Brazilian Serie B": (7, "Roughly equivalent to USL Championship level"),
    "Polish Ekstraklasa": (6, "Slightly below USL Championship"),
    "Norwegian Eliteserien": (5, "Comparable to USL League One"),
    "Argentine Primera División": (8, "Above USL Championship, approaching MLS level"),
    "Colombian Liga BetPlay": (6, "Slightly below USL Championship"),
    "Mexican Liga MX": (8, "Above USL Championship, approaching MLS level"),
    "Japanese J-League": (6, "Slightly below USL Championship"),
    "Korean K-League": (6, "Slightly below USL Championship"),
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


@st.cache_resource
def get_anthropic_client() -> Anthropic:
    return Anthropic(api_key=ANTHROPIC_API_KEY)


@st.cache_data(ttl=3600, show_spinner=False)
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


TARGET_COLUMNS = [
    "name", "age", "club", "goals", "assists", "appearances", "market_value_euros", "usl_fit_score",
]

SORT_OPTIONS = {
    "USL fit score": ("usl_fit_score", False),
    "Goals": ("goals", False),
    "Age": ("age", True),
}


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


def enrich_player(player_name: str, league: str, target_level: str) -> tuple:
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
    except Exception as e:
        print(f"[enrich_player] {player_name}: API call failed: {e}")
        return (f"<exception: {e}>", None)

    print(f"[enrich_player] {player_name} RAW:\n{text}\n---")
    data = _extract_json_object(text)
    if data is None:
        print(f"[enrich_player] {player_name}: JSON parse failed")
        return (text, None)
    if not data.get("name"):
        data["name"] = player_name
    print(f"[enrich_player] {player_name} PARSED: {data}")
    return (text, data)


def build_targets(league: str, position: str, target_level: str) -> tuple:
    debug = {"raw_listing": "", "parsed_names": [], "enrichments": [], "error": None}
    try:
        listing = discover_players(league, position, target_level)
    except Exception as e:
        debug["error"] = f"discover_players failed: {e}"
        print(f"[build_targets] discover_players failed: {e}")
        return [], debug
    debug["raw_listing"] = listing
    print(f"[build_targets] RAW LISTING:\n{listing}\n---")

    names = parse_player_names(listing)
    debug["parsed_names"] = names
    print(f"[build_targets] PARSED NAMES: {names}")
    if not names:
        return [], debug

    with ThreadPoolExecutor(max_workers=5) as ex:
        results = list(ex.map(lambda n: (n, *enrich_player(n, league, target_level)), names))

    rows = []
    for name, raw, parsed in results:
        debug["enrichments"].append({"name": name, "raw": raw, "parsed": parsed})
        if parsed:
            rows.append(parsed)
    print(f"[build_targets] FINAL ROWS: {len(rows)}")
    return rows, debug


def _latin1(text: str) -> str:
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^\s*[-*]\s+", "  • ", text, flags=re.MULTILINE)
    return text


def build_report_pdf(player_name: str, league: str, verdict: Optional[str], report: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(0, 10, _latin1(player_name))
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, _latin1(f"League: {league}"), new_x="LMARGIN", new_y="NEXT")
    if verdict:
        pdf.cell(0, 7, _latin1(f"Verdict: {verdict}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 11)
    body = _latin1(_strip_markdown(report))
    pdf.multi_cell(0, 6, body)

    return bytes(pdf.output())


def build_email_body(entry: dict) -> str:
    verdict = entry.get("verdict") or "N/A"
    header = (
        f"Player: {entry['player_name']}\n"
        f"League: {entry['league']}\n"
        f"Verdict: {verdict}\n"
        + "-" * 50 + "\n\n"
    )
    body = _strip_markdown(entry["report"])
    footer = "\n\n----\nGenerated by ScoutAI"
    return header + body + footer


def send_report_email(recipient: str, subject: str, body: str) -> None:
    sender = os.environ.get("GMAIL_SENDER", "").strip()
    password = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if not sender or not password:
        raise RuntimeError("Email not configured. Set GMAIL_SENDER and GMAIL_APP_PASSWORD in .env.")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as server:
        server.login(sender, password)
        server.send_message(msg)


def render_report_view(entry: dict) -> None:
    if entry.get("source_badge"):
        kind, text = entry["source_badge"]
        if kind == "success":
            st.success(text)
        else:
            st.caption(text)

    verdict = entry.get("verdict")
    league_name = entry["league"]
    report = entry["report"]
    player = entry["player_name"]

    if verdict:
        with st.container(border=True):
            if verdict == "SIGN":
                st.success("### Verdict: SIGN ✅")
            elif verdict == "MONITOR":
                st.warning("### Verdict: MONITOR ⚠️")
            elif verdict == "PASS":
                st.error("### Verdict: PASS ❌")

    score, label = LEAGUE_QUALITY[league_name]
    with st.container(border=True):
        st.subheader("League Quality vs USL")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(label=league_name, value=f"{score}/10")
        with col2:
            st.progress(score / 10)
            st.caption(label)

    with st.container(border=True):
        st.markdown(report)

    pdf_bytes = build_report_pdf(player, league_name, verdict, report)
    safe_slug = re.sub(r"[^A-Za-z0-9]+", "_", player.strip()) or "report"
    idx = st.session_state.get("selected_idx", 0)
    st.download_button(
        "Download Report as PDF",
        data=pdf_bytes,
        file_name=f"scoutai_{safe_slug}.pdf",
        mime="application/pdf",
        key=f"download_pdf_{idx}",
    )

    with st.expander("Send Report"):
        recipient = st.text_input(
            "Recipient email",
            key=f"email_recipient_{idx}",
            placeholder="scout@club.com",
        )
        if st.button("Send", key=f"email_send_{idx}"):
            recipient_clean = recipient.strip()
            if not recipient_clean or not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", recipient_clean):
                st.error("Enter a valid email address.")
            else:
                try:
                    subject = f"ScoutAI Report — {player} ({league_name})"
                    body = build_email_body(entry)
                    send_report_email(recipient_clean, subject, body)
                    st.success(f"Sent to {recipient_clean}")
                except Exception as e:
                    st.error(f"Send failed: {e}")


st.set_page_config(page_title="ScoutAI", page_icon=None, layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

VERDICT_BADGES = {"SIGN": "✅", "MONITOR": "⚠️", "PASS": "❌"}

CUSTOM_CSS = """
<style>
.stApp {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 820px;
}

.scoutai-title {
    background: linear-gradient(90deg, #22c55e 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin: 0 0 0.35rem 0;
}

.scoutai-accent {
    height: 2px;
    width: 72px;
    background: linear-gradient(90deg, #22c55e 0%, transparent 100%);
    border: 0;
    margin: 0 0 1.25rem 0;
}

.scoutai-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0 0 1.75rem 0;
}

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: #ffffff;
    border: 0;
    padding: 0.75rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 600;
    border-radius: 8px;
    width: 100%;
    box-shadow: 0 2px 10px rgba(34, 197, 94, 0.18);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(34, 197, 94, 0.3);
    color: #ffffff;
    border: 0;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #161b22;
    border: 1px solid #1f2937 !important;
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid #1f2937;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    padding: 0.5rem 1rem;
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #161b22 !important;
    border: 1px solid #1f2937 !important;
}

h2, h3 {
    margin-top: 1.25rem !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.01em;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 0.85rem;
}
[data-testid="stMetricValue"] {
    font-weight: 700;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    "<h1 class='scoutai-title'>ScoutAI</h1>"
    "<hr class='scoutai-accent' />"
    "<p class='scoutai-subtitle'>International soccer scouting, powered by Claude.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Scouting History")
    if not st.session_state.history:
        st.caption("No reports yet. Generate one to start building history.")
    else:
        for i, entry in enumerate(st.session_state.history):
            badge = VERDICT_BADGES.get(entry.get("verdict"), "•")
            label = f"{badge}  {entry['player_name']}  —  {entry['league']}"
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.selected_idx = i

tab_report, tab_discovery = st.tabs(["Scouting Report", "Find Players"])

with tab_report:
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
                    known_key = player_name.strip().lower()
                    source_badge = None
                    if known_key in KNOWN_PLAYERS:
                        source_badge = ("success", "✓ Verified player data")
                        known = format_known_player(KNOWN_PLAYERS[known_key])
                        combined_stats = (
                            f"[Verified player data]\n{known}\n\n[Additional notes]\n{stats.strip()}"
                            if stats.strip()
                            else f"[Verified player data]\n{known}"
                        )
                    else:
                        combined_stats = stats.strip()
                    report = generate_report(player_name.strip(), league, combined_stats)
                    verdict = extract_verdict(report)
                    entry = {
                        "player_name": player_name.strip(),
                        "league": league,
                        "verdict": verdict,
                        "report": report,
                        "source_badge": source_badge,
                    }
                    st.session_state.history.append(entry)
                    st.session_state.selected_idx = len(st.session_state.history) - 1
                except Exception as e:
                    st.error(f"Request failed: {e}")

    selected_idx = st.session_state.get("selected_idx")
    if (
        selected_idx is not None
        and 0 <= selected_idx < len(st.session_state.history)
    ):
        render_report_view(st.session_state.history[selected_idx])

with tab_discovery:
    st.subheader("Player Discovery")
    st.caption("Surface transfer targets by league, position, and destination tier.")

    discovery_league = st.selectbox("League", DISCOVERY_LEAGUES, key="discovery_league")
    position = st.selectbox("Position", ["Striker", "Midfielder", "Defender", "Goalkeeper"])
    target_level = st.selectbox("Target level", ["MLS", "USL Championship", "USL League One"])

    if st.button("Find Players", type="primary", key="discovery_button"):
        if not ANTHROPIC_API_KEY:
            st.error("ANTHROPIC_API_KEY is not set. Add it at the top of app.py.")
        else:
            with st.spinner("Scouting candidates..."):
                try:
                    rows, debug = build_targets(discovery_league, position, target_level)
                    st.session_state.discovery_rows = rows
                    st.session_state.discovery_debug = debug
                    st.session_state.discovery_meta = {
                        "league": discovery_league,
                        "position": position,
                        "target_level": target_level,
                    }
                except Exception as e:
                    st.error(f"Request failed: {e}")
                    st.session_state.discovery_rows = []
                    st.session_state.discovery_debug = {"error": str(e)}

    rows = st.session_state.get("discovery_rows")
    if rows:
        meta = st.session_state.get("discovery_meta", {})
        meta_league = meta.get("league", discovery_league)
        d_score, d_label = LEAGUE_QUALITY[meta_league]

        st.subheader("Recommended Transfer Targets")
        st.caption("Powered by Claude Scout Intelligence")

        with st.container(border=True):
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            meta_col1.metric("League", meta_league)
            meta_col2.metric("Position", meta.get("position", position))
            meta_col3.metric("Target", meta.get("target_level", target_level))
            st.progress(d_score / 10)
            st.caption(f"{meta_league}: {d_label}")

        sort_choice = st.selectbox("Sort by", list(SORT_OPTIONS.keys()), key="discovery_sort")
        sort_col, ascending = SORT_OPTIONS[sort_choice]

        df = pd.DataFrame(rows)
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[TARGET_COLUMNS]
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

        with st.container(border=True):
            st.dataframe(df, use_container_width=True, hide_index=True)
    elif "discovery_rows" in st.session_state:
        st.warning("Couldn't extract any candidates. Try a different combination.")
        debug = st.session_state.get("discovery_debug", {})
        with st.expander("Debug: what Claude returned", expanded=True):
            if debug.get("error"):
                st.error(debug["error"])
            st.markdown("**Parsed names:**")
            st.write(debug.get("parsed_names", []))
            st.markdown("**Raw player listing (first Claude call):**")
            st.code(debug.get("raw_listing", "") or "<empty>")
            enrichments = debug.get("enrichments", [])
            if enrichments:
                st.markdown("**Per-player enrichment responses:**")
                for entry in enrichments:
                    st.markdown(f"- `{entry['name']}` → parsed: `{entry['parsed'] is not None}`")
                    st.code(entry.get("raw", "") or "<empty>")
