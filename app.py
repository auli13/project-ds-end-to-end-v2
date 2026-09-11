import streamlit as st
import json
import re
import time
import pandas as pd
from google import genai
from functions_api import ALL_TOOLS_FUNCTIONS, get_risk_at_step, get_step_from_datetime

st.set_page_config(page_title="Factory Predictive Maintenance", page_icon="🏭", layout="wide")

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
MODEL = "gemini-3.7-flash"

TOOLS_SCHEMA = [
    {"type": "function", "name": "predict_station_risk",
     "description": "Get the probability of a Mechanical/Electrical failure in the next 30 minutes for a specific station.",
     "parameters": {"type": "object", "properties": {"station": {"type": "string"}}, "required": ["station"]}},
    {"type": "function", "name": "get_prob_fail_sensor",
     "description": ("Estimate which sensor is closest to its known failure threshold right now for a station "
                      "(a similarity heuristic, not a trained classifier). Useful when asked which sensor is "
                      "likely to cause the next failure."),
     "parameters": {"type": "object", "properties": {"station": {"type": "string"}}, "required": ["station"]}},
    {"type": "function", "name": "get_station_sensors",
     "description": "Get the list of sensors relevant to a specific station (which sensors exist/matter there).",
     "parameters": {"type": "object", "properties": {"station": {"type": "string"}}, "required": ["station"]}},
    {"type": "function", "name": "get_sensor_threshold",
     "description": "Get the detected failure threshold for a specific sensor at a specific station.",
     "parameters": {"type": "object", "properties": {
         "station": {"type": "string"}, "sensor": {"type": "string"}}, "required": ["station", "sensor"]}},
    {"type": "function", "name": "get_downtime_history",
     "description": ("Get downtime statistics for a station. IMPORTANT: when discussing Mechanical or "
                      "Electrical failure risk, always pass cause='Mechanical' or cause='Electrical' — "
                      "never leave it empty in that context, since the unfiltered total mixes in "
                      "process stoppages (Blocked, Starved, Changeover, Quality) unrelated to equipment failure."),
     "parameters": {"type": "object", "properties": {
         "station": {"type": "string"}, "cause": {"type": "string"}}, "required": ["station"]}},
    {"type": "function", "name": "get_model_performance",
     "description": "Get the evaluation metrics (recall, precision, etc.) of the model used for a station.",
     "parameters": {"type": "object", "properties": {"station": {"type": "string"}}, "required": ["station"]}},
]


def _is_rate_limit_error(exc) -> bool:
    msg = str(exc)
    return "429" in msg or "RateLimitError" in type(exc).__name__ or "quota" in msg.lower()


def _seconds_to_wait(exc, default=15) -> float:
    """Parse Gemini's 'Please retry in 39.29s' hint from the error message, if present."""
    match = re.search(r"retry in ([\d.]+)s", str(exc))
    return float(match.group(1)) + 1 if match else default


def _create_with_retry(max_retries=1, **kwargs):
    """Wraps client.interactions.create with a single automatic retry on rate-limit errors,
    waiting for the exact time Gemini itself suggests in the error message."""
    for attempt in range(max_retries + 1):
        try:
            return client.interactions.create(**kwargs)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < max_retries:
                time.sleep(_seconds_to_wait(e))
                continue
            raise


def run_agent(user_message, previous_id=None):
    interaction = _create_with_retry(
        model=MODEL, input=user_message, tools=TOOLS_SCHEMA, previous_interaction_id=previous_id,
    )
    fc_steps = [s for s in interaction.steps if s.type == "function_call"]
    while fc_steps:
        function_results = []
        for step in fc_steps:
            func = ALL_TOOLS_FUNCTIONS[step.name]
            result_data = func(**step.arguments)
            function_results.append({
                "type": "function_result", "name": step.name, "call_id": step.id,
                "result": [{"type": "text", "text": json.dumps(result_data)}],
            })
        interaction = _create_with_retry(
            model=MODEL, input=function_results, tools=TOOLS_SCHEMA, previous_interaction_id=interaction.id,
        )
        fc_steps = [s for s in interaction.steps if s.type == "function_call"]
    return interaction.output_text, interaction.id


# HEADER
st.title("🏭 Factory Predictive Maintenance Assistant")
st.caption("Predicting Mechanical/Electrical failures 30 minutes in advance for each ST")

# ============ DASHBOARD ============
from functions_api import result  # dict loaded in functions_api.py

max_step = len(result['ST1']['test_df']) - 1  # all stations share the same time axis
test_start = result['ST1']['test_df']['timestamp'].iloc[0]
test_end = result['ST1']['test_df']['timestamp'].iloc[-1]

if 'sim_step' not in st.session_state:
    st.session_state.sim_step = 0

st.caption(f"Available range in the test period: {test_start} → {test_end}")

col_date, col_time, col_go = st.columns([2, 1, 1])
with col_date:
    target_date = st.date_input(
        "Date", value=test_start.date(),
        min_value=test_start.date(), max_value=test_end.date()
    )
with col_time:
    target_time = st.time_input("Time", value=test_start.time())
with col_go:
    st.markdown("<div style='height: 1.9rem;'></div>", unsafe_allow_html=True)
    if st.button("🔎 Jump to date/time", use_container_width=True):
        target_dt = pd.Timestamp.combine(target_date, target_time)
        st.session_state.sim_step = get_step_from_datetime(target_dt)

status = [get_risk_at_step(s, st.session_state.sim_step) for s in result]

st.markdown(
    f"🕒 {status[0]['timestamp']}  —  minute {st.session_state.sim_step} / {max_step}"
    f"&nbsp;&nbsp;&nbsp;&nbsp;**Simulated minute (test period, final 20%)**",
    unsafe_allow_html=True,
)

col_a, col_b, col_c = st.columns([5, 1, 1])
with col_a:
    st.session_state.sim_step = st.slider(
        "Simulated minute (test period, final 20%)", 0, max_step, st.session_state.sim_step,
        label_visibility="collapsed",
    )
with col_b:
    if st.button("◀️", use_container_width=True):
        st.session_state.sim_step = max(0, st.session_state.sim_step - 1)
with col_c:
    if st.button("▶️", use_container_width=True):
        st.session_state.sim_step = min(max_step, st.session_state.sim_step + 1)

STATION_INFO = {
    'ST1': ('Robotics', '🤖'),
    'ST2': ('CNC', '⚙️'),
    'ST3': ('Robotics', '🤖'),
    'ST4': ('Assembly', '🔩'),
    'ST5': ('Welding', '🔥'),
    'ST6': ('Cooling', '❄️'),
    'ST7': ('Inspection', '🔍'),
    'ST8': ('Packaging', '📦'),
}

STATION_ICON_IMAGES = {
    'Robotics': "images_hmi/robotics_icon.png",
    'Welding': "images_hmi/welding_icon.png",
}

st.subheader("Risk Overview")
cols = st.columns(4)
for i, row in enumerate(status):
    with cols[i % 4]:
        risk = row['risk_next_30min']
        station_type, icon = STATION_INFO.get(row['station'], ('', ''))

        with st.container(border=True, height=160):
            col_icon, col_name, col_pred, col_real, col_pct = st.columns([1, 3, 1, 1, 2])
            with col_icon:
                if station_type in STATION_ICON_IMAGES:
                    st.image(STATION_ICON_IMAGES[station_type], width=20)
                else:
                    st.markdown(f"<div style='font-size:16px'>{icon}</div>", unsafe_allow_html=True)
            with col_name:
                st.markdown(
                    f"**{row['station']}** <span style='color:gray; font-size:0.85em'>{station_type}</span>",
                    unsafe_allow_html=True,
                )
            with col_pred:
                st.markdown("⚠️" if row['predicted_failure'] else "➖")
            with col_real:
                st.markdown("🔧" if row['actual_failure_next_30min'] else "➖")
            with col_pct:
                st.markdown(f"<div style='text-align:right'><b>{risk*100:.0f}%</b></div>", unsafe_allow_html=True)

            if row['predicted_failure'] and row['prob_fail_sensor']:
                sensor_label = row['prob_fail_sensor'].replace('_', ' ').title()
                st.caption(f"🔍 Probable Sensor failure: {sensor_label}")

            if not row['is_running'] and row['stop_cause'] in ['Mechanical', 'Electrical']:
                cause_label = "Mechanical" if row['stop_cause'] == 'Mechanical' else "Electric"
                if row['trigger_sensor']:
                    sensor_label = row['trigger_sensor'].replace('_', ' ').title()
                    st.error(f"🛑 {cause_label} Fail:  \n{sensor_label} Error")
                else:
                    st.error(f"🛑 {cause_label} Fail")
            else:
                st.success(f"✅ {row['station']} Running")

# ============ ASSISTANT ============
st.divider()
st.subheader("💬 Ask the Assistant")

if "display_history" not in st.session_state:
    st.session_state.display_history = []
    st.session_state.last_interaction_id = None

with st.form("chat_form", clear_on_submit=True):
    col_input, col_send = st.columns([6, 1])
    with col_input:
        prompt = st.text_input(
            "Ask about station risk, thresholds, downtime, model reliability...",
            label_visibility="collapsed",
            placeholder="Ask about station risk, thresholds, downtime, model reliability...",
        )
    with col_send:
        submitted = st.form_submit_button("Send", use_container_width=True)

chat_box = st.container(height=260) if st.session_state.display_history else st.container()
with chat_box:
    for role, text in st.session_state.display_history:
        avatar = "🧑‍🔧" if role == "user" else "🤖"
        st.chat_message(role, avatar=avatar).write(text)

if submitted and prompt.strip():
    with chat_box:
        st.chat_message("user", avatar="🧑‍🔧").write(prompt)
    st.session_state.display_history.append(("user", prompt))

    with st.spinner("Analyzing..."):
        try:
            answer, new_id = run_agent(prompt, previous_id=st.session_state.last_interaction_id)
            if not answer or not answer.strip():
                answer = "🤔 I couldn't generate a response for that — try rephrasing the question."
                new_id = st.session_state.last_interaction_id
        except Exception as e:
            new_id = st.session_state.last_interaction_id
            if _is_rate_limit_error(e):
                answer = "⏳ The assistant hit Gemini's free-tier rate limit. Please wait a few seconds and ask again."
            else:
                answer = f"⚠️ The assistant ran into an unexpected error: {e}"
    st.session_state.last_interaction_id = new_id

    with chat_box:
        st.chat_message("assistant", avatar="🤖").write(answer)
    st.session_state.display_history.append(("assistant", answer))