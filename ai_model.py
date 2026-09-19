"""
MedFlow - AI component: a triage urgency classifier.

WHAT IT DOES
    Looks at a patient's vital signs (age, heart rate, blood pressure,
    oxygen level, breathing rate, temperature) and PREDICTS how urgent they
    are, from 1 (critical) to 5 (not urgent).

HOW IT LEARNS
    We generate thousands of example patients. Each one gets a "true" urgency
    label from medical-style danger zones + random noise (real triage is
    never perfectly clean). A RandomForest model then LEARNS the pattern from
    those examples. It trains on 80% of the data and is scored on the
    other 20% it has never seen, so the accuracy we report is honest.

WHERE IT IS USED
    simulation.py calls model.predict() for every arriving patient and the
    scheduler ranks the queue using the PREDICTED urgency - never the true
    one. So the AI output really drives the scheduling.

NOTE FOR THE README
    The training data is synthetic. If you can find a real triage dataset
    (e.g. on Kaggle), swap it in inside train_model() - nothing else changes.

Run this file on its own to see the model's accuracy:   python ai_model.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

FEATURES = ["age", "heart_rate", "systolic_bp", "spo2", "resp_rate", "temp"]


def generate_vitals(rng, n):
    """Make n patients' worth of realistic-ish vital signs."""
    age = rng.integers(1, 95, n)
    heart_rate = np.clip(rng.normal(88, 25, n), 35, 190)
    systolic_bp = np.clip(rng.normal(122, 26, n), 60, 220)
    spo2 = np.clip(rng.normal(96, 3.5, n), 70, 100)       # oxygen saturation %
    resp_rate = np.clip(rng.normal(17, 5, n), 6, 40)      # breaths per minute
    temp = np.clip(rng.normal(37.2, 0.9, n), 34, 41)      # degrees Celsius
    return np.column_stack([age, heart_rate, systolic_bp, spo2, resp_rate, temp])


def danger_points(X):
    """Add up 'danger points' from abnormal vitals (bigger = sicker)."""
    age, hr, sbp, spo2, rr, temp = X.T
    pts = np.zeros(len(X))
    pts += np.where((hr > 130) | (hr < 45), 3, np.where((hr > 110) | (hr < 55), 1, 0))
    pts += np.where((sbp < 90) | (sbp > 180), 3, np.where((sbp < 100) | (sbp > 160), 1, 0))
    pts += np.where(spo2 < 90, 3, np.where(spo2 < 94, 1, 0))
    pts += np.where((rr > 28) | (rr < 9), 3, np.where(rr > 22, 1, 0))
    pts += np.where((temp > 39.5) | (temp < 35), 2, np.where(temp > 38.3, 1, 0))
    pts += np.where((age > 75) | (age < 2), 1, 0)
    return pts


def noisy_urgency(rng, X):
    """The 'true' urgency label: danger points + noise, mapped to 1-5."""
    pts = danger_points(X) + rng.normal(0, 0.6, len(X))
    return np.select(
        [pts >= 6.5, pts >= 4.5, pts >= 2.5, pts >= 0.5],
        [1, 2, 3, 4],
        default=5,
    )


def train_model(n=8000, seed=42):
    """Train the classifier. Returns (model, metrics)."""
    rng = np.random.default_rng(seed)
    X = generate_vitals(rng, n)
    y = noisy_urgency(rng, X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(np.mean(predictions == y_test)),
        "within_one_level": float(np.mean(np.abs(predictions - y_test) <= 1)),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    return model, metrics


if __name__ == "__main__":
    _, m = train_model()
    print(f"Trained on {m['train_size']} patients, tested on {m['test_size']}")
    print(f"Exact-level accuracy:      {m['accuracy']:.1%}")
    print(f"Within 1 level of truth:   {m['within_one_level']:.1%}")
