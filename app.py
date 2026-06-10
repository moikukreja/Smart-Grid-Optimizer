"""
app.py - Smart Energy Grid Optimizer  (Streamlit dashboard)
============================================================
The interactive "Dashboard" deliverable for Topic 5.

Design goals for this version:
  * Look clean and professional (custom theme, cards, infographics).
  * Be understandable to someone with ZERO background — every chart has a
    plain-English "what am I looking at" explanation and an auto-generated
    headline insight.
  * Surface real decision-making value: RL vs optimiser, energy mix, gauges,
    a cost-emissions Pareto, and a sensitivity sweep.

Run locally :  streamlit run app.py
Deploy      :  push this repo to GitHub -> share.streamlit.io -> New app
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from model import (
    GridConfig,
    DQNAgent,
    generate_day,
    train_dqn,
    evaluate_rl_day,
    evaluate_constraint_day,
    compute_metrics,
    energy_mix,
    sensitivity_emission_cap,
)

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "dqn_weights.pth")
HISTORY_PATH = os.path.join(HERE, "train_history.json")

# ==========================================================================
# Page config + theme
# ==========================================================================
st.set_page_config(page_title="Smart Energy Grid Optimizer",
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

SRC = {  # source -> (colour, emoji, one-line plain description)
    "Solar":   ("#F9C80E", "☀️", "Free & clean — but only in daylight"),
    "Wind":    ("#43AA8B", "💨", "Clean — but gusty and unpredictable"),
    "Battery": ("#4CC9F0", "🔋", "Stores spare solar for the evening peak"),
    "Gas":     ("#577590", "🔥", "Flexible, moderate cost & emissions"),
    "Coal":    ("#9B5DE5", "🏭", "Cheap but the dirtiest source"),
    "Unmet":   ("#F94144", "⚠️", "Demand we failed to meet (a blackout)"),
}
ACCENT = "#F9C80E"

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px;}}
/* hero */
.hero {{
  background: linear-gradient(120deg, rgba(249,200,14,.16), rgba(67,170,139,.14));
  border: 1px solid rgba(255,255,255,.08); border-radius: 18px;
  padding: 22px 26px; margin-bottom: 14px;
}}
.hero h1 {{margin: 0; font-size: 2.05rem; letter-spacing:-.5px;}}
.hero p {{margin: 6px 0 0; color: #c9d1d9; font-size: 1.02rem;}}
.badge {{display:inline-block; padding:4px 12px; margin:4px 6px 0 0;
  border-radius:999px; font-size:.80rem; font-weight:600;
  background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12);}}
/* insight banner */
.insight {{background: rgba(67,170,139,.12); border-left: 5px solid {ACCENT};
  border-radius: 10px; padding: 14px 18px; margin: 6px 0 4px; font-size: 1.02rem;}}
/* concept cards */
.cgrid {{display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:6px 0 2px;}}
.ccard {{background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.09);
  border-top:3px solid var(--c); border-radius:12px; padding:14px 14px 12px;}}
.ccard .ico {{font-size:1.5rem;}}
.ccard .ti {{font-weight:700; margin:4px 0 2px; font-size:.98rem;}}
.ccard .de {{color:#aab3bd; font-size:.84rem; line-height:1.3;}}
@media (max-width: 900px) {{.cgrid {{grid-template-columns:repeat(2,1fr);}}}}
/* source legend chips */
.srcchip {{display:flex; align-items:center; gap:8px; padding:7px 10px; margin:5px 0;
  background:rgba(255,255,255,.03); border-radius:9px; font-size:.86rem;}}
.dot {{width:13px; height:13px; border-radius:4px; flex:0 0 auto;}}
.small {{color:#9aa3ad; font-size:.84rem;}}
section[data-testid="stSidebar"] {{border-right:1px solid rgba(255,255,255,.06);}}
</style>
""", unsafe_allow_html=True)


# ==========================================================================
# Train the DQN once and cache it (keeps the demo instant after first load)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def get_trained_agent(episodes: int = 350):
    base = GridConfig()
    action_dim = base.coal_levels * base.discharge_levels

    # Fast path: load pre-trained weights shipped in the repo -> instant start.
    if os.path.exists(WEIGHTS_PATH):
        agent = DQNAgent(state_dim=6, action_dim=action_dim)
        agent.q.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        agent.target.load_state_dict(agent.q.state_dict())
        agent.eps = 0.0  # act greedily
        history = []
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH) as f:
                history = json.load(f)
        return agent, history

    # Fallback: no weights found -> train once (shows a progress bar).
    bar = st.progress(0.0, text="🧠 Training the Deep Q-Network agent (one-time)…")

    def cb(ep, total, _r):
        bar.progress(ep / total, text=f"🧠 Training the Deep RL agent… {ep}/{total} episodes")

    agent, history = train_dqn(base, episodes=episodes, seed=0, progress_cb=cb)
    bar.empty()
    return agent, history


# ==========================================================================
# Sidebar — scenario presets + controls
# ==========================================================================
PRESETS = {
    "⚖️ Balanced day": dict(pen=1.0, dvar=0.10, ecap=40, seed=8),
    "☀️ Sunny & calm": dict(pen=1.35, dvar=0.05, ecap=45, seed=3),
    "☁️ Cloudy, high demand": dict(pen=0.65, dvar=0.22, ecap=40, seed=12),
    "🌍 Strict climate policy": dict(pen=1.1, dvar=0.10, ecap=20, seed=8),
}
for k, v in PRESETS["⚖️ Balanced day"].items():
    st.session_state.setdefault(k, v)


def _apply_preset():
    for k, v in PRESETS.get(st.session_state.get("preset", ""), {}).items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### 🎛️ Scenario controls")
    st.caption("Set the day, then explore the tabs. Start with a preset 👇")
    st.selectbox("Quick preset", list(PRESETS.keys()), key="preset",
                 on_change=_apply_preset)

    st.markdown("**Fine-tune the scenario**")
    pen = st.slider("☀️💨 Renewable capacity", 0.5, 1.5, step=0.05, key="pen",
                    help="How much solar + wind the city has installed. "
                         "Higher = greener but the sun/wind still come and go.")
    dvar = st.slider("🎲 Demand uncertainty", 0.0, 0.30, step=0.01, key="dvar",
                     help="How unpredictable electricity demand is. Higher = "
                          "more random swings the grid must absorb.")
    ecap = st.slider("🌍 Carbon limit (tCO₂/hour)", 15, 70, step=1, key="ecap",
                     help="The hard hourly CO₂ ceiling the optimiser must obey. "
                          "Lower = stricter climate policy.")
    seed = st.number_input("📅 Which day to simulate", 0, 9999, step=1, key="seed",
                           help="Each number is a different random day "
                                "(different weather & demand).")

    st.markdown("---")
    st.markdown("##### 🧠 Powered by")
    st.markdown(
        '<span class="badge">Deep Q-Network</span>'
        '<span class="badge">OR-Tools LP</span>'
        '<span class="badge">Monte-Carlo</span>'
        '<span class="badge">Sensitivity</span>',
        unsafe_allow_html=True)
    st.caption("An AI agent (Deep RL) vs. a math optimiser — compared live.")

# Build scenario config from the controls
cfg = GridConfig(renewable_penetration=pen, demand_variability=dvar,
                 emission_cap=float(ecap))

# Train (cached) + run both strategies on the SAME day
agent, history = get_trained_agent()
day = generate_day(cfg, seed=int(seed))
rl_df = evaluate_rl_day(cfg, day, agent)
cp_df = evaluate_constraint_day(cfg, day)
rl_m, cp_m = compute_metrics(cfg, rl_df), compute_metrics(cfg, cp_df)


# ==========================================================================
# Reusable figures
# ==========================================================================
def dispatch_chart(df, title, h=380):
    fig = go.Figure()
    for s in ["Solar", "Wind", "Battery", "Gas", "Coal", "Unmet"]:
        fig.add_trace(go.Scatter(
            x=df["hour"], y=df[s.lower()], name=s, mode="lines",
            stackgroup="one", line=dict(width=0.5, color=SRC[s][0]),
            hovertemplate=f"{s}: %{{y:.0f}} MW<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=df["hour"], y=df["demand"], name="Demand", mode="lines",
        line=dict(color="#ffffff", width=2, dash="dash"),
        hovertemplate="Demand: %{y:.0f} MW<extra></extra>"))
    fig.update_layout(
        title=title, height=h, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=44, b=10),
        xaxis_title="Hour of day", yaxis_title="Power (MW)",
        legend=dict(orientation="h", y=-0.18))
    return fig


def mix_donut(df):
    mix = energy_mix(df)
    labels = [k for k in mix if mix[k] > 1e-6]
    fig = go.Figure(go.Pie(
        labels=labels, values=[mix[k] for k in labels], hole=0.62,
        marker=dict(colors=[SRC[k][0] for k in labels]),
        textinfo="percent", sort=False,
        hovertemplate="%{label}: %{value:.0f} MWh (%{percent})<extra></extra>"))
    clean = mix["Solar"] + mix["Wind"]
    share = clean / max(sum(mix.values()), 1e-9) * 100
    fig.update_layout(
        template="plotly_dark", height=300, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
        annotations=[dict(text=f"<b>{share:.0f}%</b><br>clean", x=0.5, y=0.5,
                          font_size=20, showarrow=False)])
    return fig


def battery_chart(df, h=230):
    fig = go.Figure(go.Scatter(
        x=df["hour"], y=df["soc"], name="Charge level", mode="lines",
        fill="tozeroy", line=dict(color=SRC["Battery"][0], width=2.5),
        hovertemplate="Stored: %{y:.0f} MWh<extra></extra>"))
    fig.update_layout(
        title="🔋 Battery charge level through the day", height=h,
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Hour of day", yaxis_title="Stored (MWh)")
    return fig


def gauge(value, title, suffix="%", good_high=True):
    color = ACCENT if good_high else "#43AA8B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": suffix, "font": {"size": 26}},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": color},
               "bgcolor": "rgba(255,255,255,.05)",
               "steps": [{"range": [0, 50], "color": "rgba(249,65,68,.18)"},
                         {"range": [50, 80], "color": "rgba(249,200,14,.15)"},
                         {"range": [80, 100], "color": "rgba(67,170,139,.20)"}]}))
    fig.update_layout(template="plotly_dark", height=210,
                      paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=18, r=18, t=40, b=8))
    return fig


# ==========================================================================
# Hero + auto-generated headline insight
# ==========================================================================
st.markdown(f"""
<div class="hero">
  <h1>⚡ Smart Energy Grid Optimizer</h1>
  <p>An AI that decides — hour by hour — how to power a city from <b>sun, wind,
  gas and coal</b>, balancing <b>cost</b> 💰, <b>carbon</b> 🌍 and
  <b>keeping the lights on</b> 💡 — under real-world uncertainty.</p>
</div>
""", unsafe_allow_html=True)

# headline insight comparing the AI to the optimiser
em_save = cp_m["total_emissions"] - rl_m["total_emissions"]
em_pct = em_save / max(cp_m["total_emissions"], 1e-9) * 100
cost_diff = rl_m["total_cost"] - cp_m["total_cost"]
cost_pct = cost_diff / max(cp_m["total_cost"], 1e-9) * 100
greener = em_save >= 0
if greener:
    msg = (f"🤖 <b>The AI agent kept the lights on "
           f"{rl_m['grid_stability_index']*100:.0f}% of the day</b> while emitting "
           f"<b>{abs(em_pct):.0f}% less CO₂</b> than the pure cost-optimiser — "
           f"trading about {abs(cost_pct):.0f}% {'higher' if cost_diff>0 else 'lower'} "
           f"cost for a cleaner grid. That trade-off is the whole point.")
else:
    msg = (f"🤖 <b>The cost-optimiser is cheaper</b>, but the AI agent met "
           f"{rl_m['grid_stability_index']*100:.0f}% of demand and learns a policy "
           f"that adapts to live conditions. Use the tabs to see the trade-offs.")
st.markdown(f'<div class="insight">{msg}</div>', unsafe_allow_html=True)

with st.expander("🟢 New here? Read this 30-second explainer"):
    st.markdown(
        "- A city needs **different amounts of electricity every hour** (low at "
        "night, peaks in the morning & evening).\n"
        "- It can draw from several sources: ☀️ solar and 💨 wind are clean & cheap "
        "but **only available when the weather allows**; 🔥 gas and 🏭 coal are "
        "**always available** but cost money and emit CO₂ (coal the most); and a "
        "**🔋 battery** can stockpile spare solar for later.\n"
        "- Our **AI agent (Deep Reinforcement Learning)** learns, from thousands "
        "of simulated days, *how much of each source to use each hour* so the city "
        "stays powered at low cost **and** low emissions.\n"
        "- We compare it against a classic **math optimiser** to show where "
        "learning adds value. Move the sliders on the left to change the world "
        "and watch both strategies react. 👈")


# ==========================================================================
# Tabs
# ==========================================================================
t1, t2, t3, t4 = st.tabs(
    ["📊  Live Dispatch", "⚖️  AI vs Optimiser", "📈  Sensitivity", "🧭  How it works"])

# ---- Tab 1: live dispatch -------------------------------------------------
with t1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💰 Daily cost", f"${rl_m['total_cost']:,.0f}",
              f"{cost_pct:+.0f}% vs optimiser", delta_color="inverse")
    k2.metric("🌍 CO₂ emitted", f"{rl_m['total_emissions']:,.0f} t",
              f"{-em_pct:+.0f}% vs optimiser", delta_color="inverse")
    k3.metric("🌱 Clean energy share", f"{rl_m['renewable_utilization']*100:.0f}%")
    k4.metric("💡 Lights kept on", f"{rl_m['grid_stability_index']*100:.0f}%")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(dispatch_chart(rl_df, "How the AI powered the city, hour by hour"),
                        use_container_width=True)
        st.plotly_chart(battery_chart(rl_df), use_container_width=True)
    with right:
        st.markdown("**Where the energy came from**")
        st.plotly_chart(mix_donut(rl_df), use_container_width=True)
        st.markdown('<div class="small">The centre shows the share from clean '
                    'sources (sun + wind + stored solar). 🔋 The battery soaks up '
                    'spare midday solar and releases it after dark.</div>',
                    unsafe_allow_html=True)

    with st.expander("❓ What am I looking at?"):
        st.markdown(
            "Each coloured band is one energy source; stacked together they must "
            "reach the **white dashed line (demand)** every hour. Notice ☀️ solar "
            "fills the **midday** belly while 🔥 gas and 🏭 coal cover the dark "
            "**morning & evening peaks**. The clever part: the **🔋 battery** "
            "charges up on surplus midday sun (watch the charge level climb), then "
            "**discharges into the evening peak** — shrinking the dirty 🏭 coal band "
            "without ever falling short of demand.")

    st.markdown("**Your energy sources**")
    cols = st.columns(6)
    for col, s in zip(cols, SRC):
        c, emo, desc = SRC[s]
        col.markdown(
            f'<div class="srcchip"><span class="dot" style="background:{c}"></span>'
            f'<span><b>{emo} {s}</b><br><span class="small">{desc}</span></span></div>',
            unsafe_allow_html=True)

# ---- Tab 2: AI vs optimiser ----------------------------------------------
with t2:
    st.markdown("#### Two ways to run the same day")
    st.markdown(
        '<div class="small">Left: our <b>AI agent</b> (Deep RL) that learned from '
        'experience. Right: a <b>math optimiser</b> that computes the cheapest legal '
        'plan each hour but can\'t see ahead. Same weather, same demand.</div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(dispatch_chart(rl_df, "🤖 AI agent (Deep RL)", h=330),
                    use_container_width=True)
    c2.plotly_chart(dispatch_chart(cp_df, "📐 Math optimiser (constraints)", h=330),
                    use_container_width=True)

    # comparison bar charts
    b1, b2 = st.columns(2)
    cost_fig = go.Figure(go.Bar(
        x=["AI agent", "Optimiser"], y=[rl_m["total_cost"], cp_m["total_cost"]],
        marker_color=[ACCENT, "#577590"], text=[f"${rl_m['total_cost']:,.0f}",
        f"${cp_m['total_cost']:,.0f}"], textposition="outside"))
    cost_fig.update_layout(title="💰 Total cost (lower is better)",
                           template="plotly_dark", height=290,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=40, b=10), yaxis_title="$")
    em_fig = go.Figure(go.Bar(
        x=["AI agent", "Optimiser"], y=[rl_m["total_emissions"], cp_m["total_emissions"]],
        marker_color=["#43AA8B", "#9B5DE5"], text=[f"{rl_m['total_emissions']:,.0f} t",
        f"{cp_m['total_emissions']:,.0f} t"], textposition="outside"))
    em_fig.update_layout(title="🌍 CO₂ emissions (lower is better)",
                         template="plotly_dark", height=290,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=10, r=10, t=40, b=10), yaxis_title="tCO₂")
    b1.plotly_chart(cost_fig, use_container_width=True)
    b2.plotly_chart(em_fig, use_container_width=True)

    # verdict
    verdict = (f"🏆 **Takeaway:** the AI emitted **{abs(em_pct):.0f}% "
               f"{'less' if greener else 'more'} CO₂** "
               f"for **{abs(cost_pct):.0f}% {'more' if cost_diff>0 else 'less'} cost**. "
               "Both share the same 🔋 battery, but the optimiser drains it greedily "
               "as soon as there's a gap, while the AI *saves* stored solar for the "
               "evening peak — that foresight is where learning pays off.")
    st.success(verdict)

    with st.expander("📋 Full metrics table"):
        comp = pd.DataFrame({
            "Metric": ["Total cost ($)", "Emissions (tCO₂)",
                       "Clean-energy share (%)", "Lights kept on (%)",
                       "CO₂ cut vs all-coal (%)"],
            "AI agent (Deep RL)": [
                f"{rl_m['total_cost']:,.0f}", f"{rl_m['total_emissions']:,.1f}",
                f"{rl_m['renewable_utilization']*100:.1f}",
                f"{rl_m['grid_stability_index']*100:.1f}",
                f"{rl_m['emission_reduction']*100:.1f}"],
            "Math optimiser": [
                f"{cp_m['total_cost']:,.0f}", f"{cp_m['total_emissions']:,.1f}",
                f"{cp_m['renewable_utilization']*100:.1f}",
                f"{cp_m['grid_stability_index']*100:.1f}",
                f"{cp_m['emission_reduction']*100:.1f}"]})
        st.table(comp)

# ---- Tab 3: sensitivity ---------------------------------------------------
with t3:
    st.markdown("#### How strict should the carbon limit be?")
    st.markdown(
        '<div class="small">We re-run the day for many carbon limits. This reveals '
        'the unavoidable tug-of-war between <b>cost</b> and <b>emissions</b> that a '
        'policy-maker must decide on.</div>', unsafe_allow_html=True)

    sens = sensitivity_emission_cap(cfg, day)

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge(rl_m["renewable_utilization"] * 100, "🌱 Clean share"),
                    use_container_width=True)
    g2.plotly_chart(gauge(rl_m["grid_stability_index"] * 100, "💡 Reliability"),
                    use_container_width=True)
    g3.plotly_chart(gauge(rl_m["emission_reduction"] * 100, "🌍 CO₂ cut vs coal"),
                    use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sens["emission_cap"], y=sens["total_cost"],
                  name="Total cost ($)", mode="lines+markers",
                  line=dict(color="#577590", width=3), yaxis="y1"))
    fig.add_trace(go.Scatter(x=sens["emission_cap"], y=sens["total_emissions"],
                  name="Emissions (tCO₂)", mode="lines+markers",
                  line=dict(color="#9B5DE5", width=3), yaxis="y2"))
    fig.add_vline(x=float(ecap), line_dash="dot", line_color=ACCENT,
                  annotation_text="your setting", annotation_position="top")
    fig.update_layout(template="plotly_dark", height=420,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="Carbon limit (tCO₂ / hour)  →  looser",
                      yaxis=dict(title="Total cost ($)", color="#577590"),
                      yaxis2=dict(title="Emissions (tCO₂)", color="#9B5DE5",
                                  overlaying="y", side="right"),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "👉 **Reading it:** loosen the limit (move right) and the grid leans on "
        "cheap 🏭 coal — **cost drops but CO₂ climbs**. Tighten it (move left) and "
        "it must switch to cleaner — pricier — power. The golden dotted line is "
        "*your* current setting. There is no free lunch: every choice picks a point "
        "on this curve.")

# ---- Tab 4: how it works --------------------------------------------------
with t4:
    st.markdown("#### The four ideas that make this work")
    cards = [
        ("Markov Decision Process", "🔗", SRC["Gas"][0],
         "Each hour is a decision: see the situation, act, get a reward, move on."),
        ("Deep Q-Network (Deep RL)", "🧠", ACCENT,
         "A neural network learns, from thousands of simulated days, which action pays off."),
        ("Constraint Programming", "📐", SRC["Wind"][0],
         "A math optimiser guarantees the plan is legal: meets demand & carbon limits."),
        ("Sensitivity Analysis", "🎲", SRC["Coal"][0],
         "We test many uncertain scenarios to map how outcomes shift with the rules."),
    ]
    html = '<div class="cgrid">' + "".join(
        f'<div class="ccard" style="--c:{c}"><div class="ico">{e}</div>'
        f'<div class="ti">{t}</div><div class="de">{d}</div></div>'
        for t, e, c, d in cards) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("#### How a single day flows")
    st.markdown(
        "```\nStochastic day  →  for each hour:  observe state  →  "
        "AI picks coal/gas split + battery discharge  →  apply limits  →  "
        "tally cost/CO₂/reliability  →  aggregate into the dashboard\n```")
    st.markdown(
        '<div class="insight">💡 <b>Innovation — smart storage.</b> A 🔋 battery '
        "charges on surplus midday solar and the agent learns <i>when</i> to "
        "release it. Because the AI plans ahead, it banks energy for the evening "
        "peak instead of spending it early like the myopic optimiser.</div>",
        unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**📦 The data**")
        st.markdown(
            "Realistic synthetic 24-hour profiles, each with random noise so every "
            "run is a fresh *uncertain* day:\n"
            "- **Demand** — a two-peak daily curve (morning + evening)\n"
            "- **Solar** — a midday bell, zero at night\n"
            "- **Wind** — gusty, slightly stronger overnight\n\n"
            "Swappable for real open data (NREL solar, ENTSO-E demand) with no "
            "model changes.")
    with cc2:
        st.markdown("**🧠 The AI's learning progress**")
        if history:
            st.line_chart(pd.DataFrame({"reward per day": history}), height=210)
            st.markdown('<div class="small">Reward rises as the agent learns — '
                        'higher means cheaper, cleaner, more reliable days.</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Pre-trained weights are loaded, so the app starts instantly. "
                    "Run `train_colab.ipynb` to regenerate the learning curve.")

    st.caption("Topic 5 · Reasoning and Decision Making Under Uncertainty · "
               "Deep RL + Constraint Programming")
