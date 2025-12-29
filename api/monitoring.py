import random
import time
from prometheus_client import Counter, Gauge

# ==========================================
# 1. METRICS DEFINITION
# ==========================================

# Counter: Counts total requests
PREDICTION_COUNTER = Counter(
    'churn_prediction_requests_total', 
    'Total number of prediction requests'
)

# Gauge: Tracks the output probability (to see if model tends to predict high/low)
CHURN_PROBABILITY_GAUGE = Gauge(
    'last_churn_probability', 
    'Probability of the last prediction'
)

# Gauge: Simulated Drift
DATA_DRIFT_GAUGE = Gauge(
    'simulated_data_drift_score', 
    'Simulated Drift Score (Random Walk)'
)

# ==========================================
# 2. DRIFT SIMULATION LOGIC
# ==========================================
def simulate_drift():
    """
    Simulates a data drift calculation.
    In a real system, this would compare current data distribution vs training data.
    Here, we simulate a random walk drift between 0 and 1.
    """
    # Simulate a value that changes slightly every time (Random Walk)
    # We use time.time() to seed it slightly or just random
    current_drift = random.random() # Returns float 0.0 to 1.0
    
    # Update the Prometheus Gauge
    DATA_DRIFT_GAUGE.set(current_drift)