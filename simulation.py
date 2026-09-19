"""
MedFlow - the simulation engine.

FLOW (matches the problem statement):
    Patient arrivals -> AI priority -> Resource allocation
        -> Scheduling simulation -> Performance statistics

One loop step = one simulated minute.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ai_model import generate_vitals, noisy_urgency

STRATEGIES = ["FCFS", "Urgency only", "Urgency + waiting time"]
RESOURCES = ["bed", "icu", "doctor", "nurse"]

# Minutes of treatment (min, max) and chance of needing an ICU, by TRUE urgency
TREATMENT_RANGE = {1: (40, 90), 2: (30, 70), 3: (20, 50), 4: (15, 35), 5: (10, 25)}
ICU_CHANCE = {1: 0.6, 2: 0.2, 3: 0.0, 4: 0.0, 5: 0.0}

# When the special events happen (minute of the day, 0-1440)
SURGE_WINDOW = (600, 720)            # 10:00 - 12:00, arrivals x3
STAFF_SHORTAGE_WINDOW = (720, 1080)  # 12:00 - 18:00, half the doctors/nurses
FAILURE_WINDOW = (360, 600)          # 06:00 - 10:00, 1 ICU bed + 3 beds down

SLA_CRITICAL_WAIT = 10   # a critical patient (urgency 1) should wait < 10 min
STARVATION_WAIT = 240    # waiting more than 4 hours counts as starvation


@dataclass
class Config:
    minutes: int = 1440              # simulate one full day
    arrival_rate: float = 0.12       # average patients per minute
    beds: int = 8
    icu: int = 2
    doctors: int = 5
    nurses: int = 7
    strategy: str = "Urgency + waiting time"
    aging_weight: float = 0.15       # points gained per minute of waiting
    surge: bool = False
    staff_shortage: bool = False
    equipment_failure: bool = False
    seed: int = 1


@dataclass
class Patient:
    pid: int
    arrival_time: int
    true_urgency: int         # hidden ground truth (1 = critical ... 5 = mild)
    predicted_urgency: int    # what the AI thinks - the scheduler only sees this
    needs_icu: bool
    treatment_time: int
    start_time: int = -1      # -1 = not started yet


def needs_of(patient):
    """Everything a patient needs at the same time (all or nothing)."""
    bed_type = "icu" if patient.needs_icu else "bed"
    return {bed_type: 1, "doctor": 1, "nurse": 1}


def base_capacity(cfg):
    return {"bed": cfg.beds, "icu": cfg.icu, "doctor": cfg.doctors, "nurse": cfg.nurses}


def capacity_at(cfg, now):
    """Resource limits at a given minute (events can reduce them)."""
    cap = base_capacity(cfg)
    if cfg.staff_shortage and STAFF_SHORTAGE_WINDOW[0] <= now < STAFF_SHORTAGE_WINDOW[1]:
        cap["doctor"] = max(1, cfg.doctors // 2)
        cap["nurse"] = max(1, cfg.nurses // 2)
    if cfg.equipment_failure and FAILURE_WINDOW[0] <= now < FAILURE_WINDOW[1]:
        cap["icu"] = max(0, cfg.icu - 1)
        cap["bed"] = max(1, cfg.beds - 3)
    return cap


# ------------------------------------------------------------
# STEP 1 + 2: patient arrivals, with the AI predicting urgency
# ------------------------------------------------------------
def generate_patients(cfg, model):
    """
    Create the day's patients. Arrivals follow a Poisson process whose rate
    changes over the day (busier around midday) and jumps during a surge.
    The AI model predicts every patient's urgency from their vitals.
    """
    rng = np.random.default_rng(cfg.seed)

    arrival_times = []
    for now in range(cfg.minutes):
        rate = cfg.arrival_rate * (1 + 0.5 * math.sin(2 * math.pi * (now / 1440 - 0.3)))
        if cfg.surge and SURGE_WINDOW[0] <= now < SURGE_WINDOW[1]:
            rate *= 3
        arrival_times += [now] * int(rng.poisson(rate))

    n = len(arrival_times)
    if n == 0:
        return []

    vitals = generate_vitals(rng, n)
    true_urgency = noisy_urgency(rng, vitals)
    predicted = model.predict(vitals)          # <-- the AI at work

    patients = []
    for i in range(n):
        u = int(true_urgency[i])
        low, high = TREATMENT_RANGE[u]
        patients.append(
            Patient(
                pid=i + 1,
                arrival_time=arrival_times[i],
                true_urgency=u,
                predicted_urgency=int(predicted[i]),
                needs_icu=bool(rng.random() < ICU_CHANCE[u]),
                treatment_time=int(rng.integers(low, high + 1)),
            )
        )
    return patients


# ------------------------------------------------------------
# STEP 2: priority score for each strategy (higher = treated sooner)
# ------------------------------------------------------------
def priority(patient, now, cfg):
    if cfg.strategy == "FCFS":
        return -patient.arrival_time                     # earlier arrival wins

    urgency_points = (6 - patient.predicted_urgency) * 10   # 1 -> 50 ... 5 -> 10
    tie_break = -patient.arrival_time * 1e-6                 # equal score: first come first

    if cfg.strategy == "Urgency only":
        return urgency_points + tie_break

    # "Urgency + waiting time": waiting adds points, so nobody starves
    waiting = now - patient.arrival_time
    return urgency_points + cfg.aging_weight * waiting + tie_break


# ------------------------------------------------------------
# STEPS 3 + 4: allocate resources and run the clock
# ------------------------------------------------------------
def run_simulation(cfg, patients):
    """Run one day with one strategy. Returns history, patients and summary."""
    patients = [replace(p, start_time=-1) for p in patients]   # fresh copies
    arrivals_by_minute = {}
    for p in patients:
        arrivals_by_minute.setdefault(p.arrival_time, []).append(p)

    base = base_capacity(cfg)
    in_use = {r: 0 for r in RESOURCES}
    waiting, treating = [], []
    rows = []

    for now in range(cfg.minutes):
        cap = capacity_at(cfg, now)

        # a) new patients join the queue
        waiting.extend(arrivals_by_minute.get(now, []))

        # b) finished patients leave and free their resources
        for p in treating[:]:
            if now - p.start_time >= p.treatment_time:
                for r, n in needs_of(p).items():
                    in_use[r] -= n
                treating.remove(p)

        # c) rank the queue, then serve whoever we can.
        #    If the top patient can't be served (e.g. no ICU bed free) we try
        #    the next one, so one blocked patient doesn't freeze the queue.
        waiting.sort(key=lambda p: priority(p, now, cfg), reverse=True)
        for p in waiting[:]:
            need = needs_of(p)
            if all(cap[r] - in_use[r] >= n for r, n in need.items()):
                for r, n in need.items():
                    in_use[r] += n
                p.start_time = now
                waiting.remove(p)
                treating.append(p)

        # d) record this minute for the dashboard
        row = {"minute": now, "queue_length": len(waiting)}
        for r in RESOURCES:
            row[r] = 100 * in_use[r] / base[r]
        rows.append(row)

    history = pd.DataFrame(rows)
    return {
        "history": history,
        "patients": patients,
        "summary": summarize(cfg, patients, history),
    }


# ------------------------------------------------------------
# STEP 5: performance statistics
# ------------------------------------------------------------
def summarize(cfg, patients, history):
    if not patients:
        return {}

    # Patients still waiting at the end count as waiting until the end
    waits = [
        (p.start_time if p.start_time >= 0 else cfg.minutes) - p.arrival_time
        for p in patients
    ]
    finished = sum(
        1 for p in patients
        if p.start_time >= 0 and p.start_time + p.treatment_time < cfg.minutes
    )
    still_waiting = sum(1 for p in patients if p.start_time < 0)

    wait_by_urgency = {}
    for u in range(1, 6):
        w = [wt for wt, p in zip(waits, patients) if p.true_urgency == u]
        wait_by_urgency[u] = float(np.mean(w)) if w else float("nan")

    critical_breaches = sum(
        1 for wt, p in zip(waits, patients)
        if p.true_urgency == 1 and wt > SLA_CRITICAL_WAIT
    )
    starved = sum(1 for wt in waits if wt > STARVATION_WAIT)

    # Little's Law sanity check: average queue length = arrival rate x average wait
    avg_queue = float(history["queue_length"].mean())
    little_predicted = (len(patients) / cfg.minutes) * float(np.mean(waits))

    return {
        "arrived": len(patients),
        "treated": finished,
        "still_waiting": still_waiting,
        "avg_wait": float(np.mean(waits)),
        "p95_wait": float(np.percentile(waits, 95)),
        "max_wait": int(max(waits)),
        "wait_by_urgency": wait_by_urgency,
        "critical_sla_breaches": critical_breaches,
        "starvation_count": starved,
        "utilization": {r: float(history[r].mean()) for r in RESOURCES},
        "ai_accuracy": float(np.mean([p.predicted_urgency == p.true_urgency for p in patients])),
        "avg_queue_length": avg_queue,
        "little_law_prediction": little_predicted,
    }


def compare_strategies(cfg, patients):
    """Run every strategy on the SAME patients so the comparison is fair."""
    table = []
    for strategy in STRATEGIES:
        s = run_simulation(replace(cfg, strategy=strategy), patients)["summary"]
        table.append({
            "Strategy": strategy,
            "Avg wait (min)": round(s["avg_wait"], 1),
            "95th pct wait (min)": round(s["p95_wait"], 1),
            "Critical patients wait (min)": round(s["wait_by_urgency"][1], 1),
            "Mildest patients wait (min)": round(s["wait_by_urgency"][5], 1),
            "Critical SLA breaches": s["critical_sla_breaches"],
            "Starved (>4h)": s["starvation_count"],
            "Treated": s["treated"],
        })
    return pd.DataFrame(table)


if __name__ == "__main__":
    # Quick test without the dashboard:  python simulation.py
    from ai_model import train_model

    model, metrics = train_model()
    print(f"AI accuracy on test set: {metrics['accuracy']:.1%}\n")
    cfg = Config()
    patients = generate_patients(cfg, model)
    print(compare_strategies(cfg, patients).to_string(index=False))
