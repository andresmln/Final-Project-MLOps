import random
import time
from prometheus_client import Counter, Gauge, Histogram

# ==========================================
# 1. METRICS DEFINITION
# ==========================================

# Counter: Counts total requests
PREDICTION_COUNTER = Counter(
    'churn_prediction_requests_total', 
    'Total number of prediction requests'
)

# Gauge: Tracks the output probability
CHURN_PROBABILITY_GAUGE = Gauge(
    'last_churn_probability', 
    'Probability of the last prediction'
)

# Gauge: Simulated Drift
DATA_DRIFT_GAUGE = Gauge(
    'simulated_data_drift_score', 
    'Simulated Drift Score (Random Walk)'
)

# Histogram: Tracks API/Model Latency
PREDICTION_LATENCY = Histogram(
    'process_request_seconds', 
    'Time spent processing prediction'
)

# ==========================================
# 2. DRIFT SIMULATION LOGIC
# ==========================================
_current_drift_value = 0.0

def simulate_drift():
    """
    Simulates a RANDOM WALK drift.
    Instead of jumping wildly, it moves slightly up or down from the last value.
    This creates a smooth, realistic-looking curve.
    """
    global _current_drift_value
    
    # 1. Randomly decide to go up or down by a small amount (e.g., -0.05 to +0.05)
    step = random.uniform(-0.05, 0.05)
    
    # 2. Update the value
    _current_drift_value += step
    
    # 3. Keep it strictly between 0.0 and 1.0 (Clamp)
    _current_drift_value = max(0.0, min(1.0, _current_drift_value))
    
    # 4. Set the Gauge
    DATA_DRIFT_GAUGE.set(_current_drift_value)