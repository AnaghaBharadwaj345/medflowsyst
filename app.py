"""
MedFlow - operations dashboard.

Run with:   streamlit run app.py

Change anything in the sidebar and the whole simulated day re-runs instantly.
"""

import pandas as pd
import streamlit as st

from ai_model import FEATURES, train_model
from simulation import (
    RESOURCES,
    STRATEGIES,
    Config,
    compare_strategies,
    generate_patients,
    run_simulation,
)

st.set_page_config(page_title="MedFlow", page_icon="🏥", layout="wide")


@st.cache_resource
def load_model():
    """Train the AI triage model once and reuse it."""
    return train_model()


model, ai_metrics = load_model()

st.title("🏥 MedFlow")
st.caption("Prioritize Patients. Optimize Resources.")

# ------------------------------------------------------------
# Sidebar: everything the user can change
# ------------------------------------------------------------
st.sidebar.header("Hospital resources")
beds = st.sidebar.slider("Beds", 2, 30, 8)
icu = st.sidebar.slider("ICU beds", 1, 10, 2)
doctors = st.sidebar.slider("Doctors", 1, 12, 5)
nurses = st.sidebar.slider("Nurses", 1, 20, 7)

st.sidebar.header("Patients")
per_hour = st.sidebar.slider("Average arrivals per hour", 1.0, 30.0, 7.2, 0.5)

st.sidebar.header("Scheduling strategy")
strategy = st.sidebar.selectbox("Strategy", STRATEGIES, index=2)
aging = 0.15
if strategy == "Urgency + waiting time":
    aging = st.sidebar.slider(
        "Waiting-time weight (points per minute)", 0.0, 1.0, 0.15, 0.05,
        help="Higher = long-waiting patients overtake urgent ones sooner.",
    )

st.sidebar.header("Emergency events")
surge = st.sidebar.checkbox("Patient surge (10:00-12:00, arrivals x3)")
shortage = st.sidebar.checkbox("Staff shortage (12:00-18:00, half the doctors and nurses)")
failure = st.sidebar.checkbox("Equipment failure (06:00-10:00, 1 ICU bed + 3 beds down)")
seed = st.sidebar.number_input("Random seed", 1, 9999, 1)

cfg = Config(
    arrival_rate=per_hour / 60,
    beds=beds, icu=icu, doctors=doctors, nurses=nurses,
    strategy=strategy, aging_weight=aging,
    surge=surge, staff_shortage=shortage, equipment_failure=failure,
    seed=int(seed),
)

# ------------------------------------------------------------
# Run the simulation: arrivals -> AI priority -> allocation -> stats
# ------------------------------------------------------------
patients = generate_patients(cfg, model)
if not patients:
    st.warning("No patients arrived. Increase the arrival rate.")
    st.stop()

result = run_simulation(cfg, patients)
s = result["summary"]
hist = result["history"].copy()
hist["hour"] = hist["minute"] / 60

tab_dash, tab_compare, tab_ai, tab_how = st.tabs(
    ["📊 Dashboard", "⚖️ Compare strategies", "🤖 AI model", "ℹ️ How it works"]
)

# ------------------------------------------------------------
# Tab 1: live operations dashboard
# ------------------------------------------------------------
with tab_dash:
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients arrived", s["arrived"])
    c2.metric("Treated", s["treated"])
    c3.metric("Still waiting at end of day", s["still_waiting"])

    c4, c5, c6 = st.columns(3)
    c4.metric("Average wait", f"{s['avg_wait']:.0f} min")
    c5.metric("95th percentile wait", f"{s['p95_wait']:.0f} min")
    c6.metric("Critical patients waiting >10 min", s["critical_sla_breaches"])

    st.subheader("Queue length through the day")
    st.line_chart(hist.set_index("hour")["queue_length"])

    st.subheader("Resource utilization through the day (%)")
    util_over_time = hist.set_index("hour")[RESOURCES].rolling(15, min_periods=1).mean()
    util_over_time.columns = ["Beds", "ICU beds", "Doctors", "Nurses"]
    st.line_chart(util_over_time)

    left, right = st.columns(2)
    with left:
        st.subheader("Average utilization (%)")
        st.bar_chart(pd.Series({
            "Beds": s["utilization"]["bed"],
            "ICU beds": s["utilization"]["icu"],
            "Doctors": s["utilization"]["doctor"],
            "Nurses": s["utilization"]["nurse"],
        }))
    with right:
        st.subheader("Average wait by true urgency (min)")
        st.bar_chart(pd.Series({
            f"Urgency {u}": w for u, w in s["wait_by_urgency"].items()
        }))
        st.caption("Urgency 1 = critical, 5 = mild.")

    with st.expander("Model check: Little's Law"):
        st.write(
            f"Average queue length in the simulation: **{s['avg_queue_length']:.2f}**  \n"
            f"Arrival rate x average wait: **{s['little_law_prediction']:.2f}**"
        )
        st.caption(
            "These two should match. If they don't, the simulator's bookkeeping "
            "has a bug (patients lost or double counted)."
        )

# ------------------------------------------------------------
# Tab 2: compare scheduling strategies on the SAME patients
# ------------------------------------------------------------
with tab_compare:
    st.subheader("Same patients, three strategies")
    comparison = compare_strategies(cfg, patients)
    st.dataframe(comparison, hide_index=True)
    st.bar_chart(
        comparison.set_index("Strategy")[
            ["Critical patients wait (min)", "Mildest patients wait (min)"]
        ]
    )
    st.markdown(
        "- **FCFS** ignores urgency: fair on arrival order, but critical patients wait too.\n"
        "- **Urgency only** protects critical patients, but mild cases can wait for hours "
        "(starvation).\n"
        "- **Urgency + waiting time** keeps critical patients fast while capping how "
        "long anyone waits.\n\n"
        "Try the surge and staff-shortage events in the sidebar to stress-test them."
    )

# ------------------------------------------------------------
# Tab 3: the AI component
# ------------------------------------------------------------
with tab_ai:
    st.subheader("AI component: triage urgency classifier")
    a1, a2, a3 = st.columns(3)
    a1.metric("Exact-level accuracy (test set)", f"{ai_metrics['accuracy']:.1%}")
    a2.metric("Within 1 level", f"{ai_metrics['within_one_level']:.1%}")
    a3.metric("Accuracy on today's patients", f"{s['ai_accuracy']:.1%}")
    st.write(
        "A Random Forest model reads each patient's vitals "
        f"({', '.join(FEATURES)}) and predicts an urgency level from 1 (critical) "
        "to 5 (mild). **The scheduler ranks the queue using this prediction, not "
        "the hidden true urgency.** So AI mistakes have real consequences, and the "
        "dashboard shows them."
    )
    st.caption(
        "Trained on synthetic patients (danger-zone rules plus noise) and scored on "
        "20% held-out data."
    )

# ------------------------------------------------------------
# Tab 4: explanation for the demo
# ------------------------------------------------------------
with tab_how:
    st.markdown(
        """
**Patient arrivals** -> patients arrive at random (busier around midday) with vital signs.

**Priority calculation** -> the AI predicts urgency; the strategy turns it into a score.

**Resource allocation** -> a patient is only treated if a bed, a doctor and a nurse are
*all* free. Capacity is never exceeded, so there are no resource conflicts.

**Scheduling simulation** -> the clock advances one minute at a time: arrivals, discharges,
re-ranking the queue, allocating.

**Performance statistics** -> waiting times, utilization, queue length and SLA breaches.
        """
    )
