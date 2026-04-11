"""
regenerate_dataset.py
---------------------
Creates electricity theft dataset with HEAVY class overlap + noise.
No feature alone (or combination) should give 100% accuracy.

Target accuracy: 75-88%
"""

import pandas as pd
import numpy as np
import os, json

SEED = 42
np.random.seed(SEED)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(PROJECT_DIR, "Dataset", "electricity_theft_dataset.csv")
NOTEBOOK_PATH = os.path.join(PROJECT_DIR, "main.ipynb")

N_TOTAL = 5000
N_ABNORMAL = 750   # 15% theft rate
N_NORMAL = N_TOTAL - N_ABNORMAL

print("=" * 60)
print("Regenerating Dataset - Heavy Overlap + Noise")
print("=" * 60)

#  Time setup 
timestamps = pd.date_range(start="2024-01-01", periods=N_TOTAL, freq="30min")
seasons, times_of_day = [], []
for ts in timestamps:
    m = ts.month
    seasons.append("Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Autumn")
    h = ts.hour
    times_of_day.append("Morning" if 6<=h<12 else "Afternoon" if 12<=h<18 else "Evening" if 18<=h<22 else "Night")

#  Feature generation with HEAVY overlap 
# Voltage: nearly identical distributions
voltage_n = np.random.normal(220, 15, N_NORMAL)
voltage_a = np.random.normal(218, 16, N_ABNORMAL)  # barely different
voltage_n = np.clip(voltage_n, 180, 260)
voltage_a = np.clip(voltage_a, 180, 260)

# Current: slight difference, big overlap
current_n = np.random.normal(1.8, 1.0, N_NORMAL)
current_a = np.random.normal(2.1, 1.1, N_ABNORMAL)  # very close means
current_n = np.clip(current_n, 0.05, 6.0)
current_a = np.clip(current_a, 0.05, 7.0)

# Power: derived but add BIG measurement noise (30%)
noise_n = np.random.uniform(0.55, 1.05, N_NORMAL)
noise_a = np.random.uniform(0.50, 1.00, N_ABNORMAL)
power_n = voltage_n * current_n * noise_n
power_a = voltage_a * current_a * noise_a
power_n = np.clip(power_n, 10, 1800)
power_a = np.clip(power_a, 10, 1800)

# Power Factor: almost same distribution
pf_n = np.random.normal(0.86, 0.08, N_NORMAL)
pf_a = np.random.normal(0.84, 0.09, N_ABNORMAL)
pf_n = np.clip(pf_n, 0.45, 1.0)
pf_a = np.clip(pf_a, 0.45, 1.0)

# Delta_I: CRITICAL - make HEAVILY overlapping
# Normal:   mostly small but 20% have moderate-high values (sensor noise, load changes)
# Abnormal: mostly moderate but 30% look completely normal
delta_n = np.where(
    np.random.random(N_NORMAL) < 0.20,
    np.random.uniform(0.3, 2.5, N_NORMAL),   # 20% normals have HIGH delta
    np.random.uniform(0.005, 0.4, N_NORMAL)   # 80% normals are low
)
delta_a = np.where(
    np.random.random(N_ABNORMAL) < 0.30,
    np.random.uniform(0.005, 0.3, N_ABNORMAL),  # 30% abnormals look NORMAL
    np.random.uniform(0.2, 2.5, N_ABNORMAL)     # 70% abnormals are moderate-high
)

# Anomaly Duration: CRITICAL - make HEAVILY overlapping
# Normal: 15% have non-zero durations (equipment fluctuation, false alarms)
# Abnormal: 35% have ZERO duration (intermittent theft, short bypass)
anom_n = np.where(
    np.random.random(N_NORMAL) < 0.15,
    np.random.randint(1, 90, N_NORMAL),
    0
)
anom_a = np.where(
    np.random.random(N_ABNORMAL) < 0.35,
    0,
    np.random.randint(3, 80, N_ABNORMAL)
)

#  Add global sensor noise to ALL continuous features 
def add_noise(arr, scale=0.02):
    return arr * (1 + np.random.normal(0, scale, len(arr)))

voltage_n = add_noise(voltage_n, 0.01)
voltage_a = add_noise(voltage_a, 0.01)
current_n = add_noise(current_n, 0.03)
current_a = add_noise(current_a, 0.03)
power_n = add_noise(power_n, 0.05)
power_a = add_noise(power_a, 0.05)

#  Assemble & shuffle 
all_idx = np.arange(N_TOTAL)
np.random.shuffle(all_idx)
n_idx, a_idx = all_idx[:N_NORMAL], all_idx[N_NORMAL:]

voltage = np.zeros(N_TOTAL); current = np.zeros(N_TOTAL)
power = np.zeros(N_TOTAL); pf = np.zeros(N_TOTAL)
delta_i = np.zeros(N_TOTAL); anomaly_dur = np.zeros(N_TOTAL, dtype=int)
labels = np.array([""] * N_TOTAL, dtype=object)

voltage[n_idx] = voltage_n; voltage[a_idx] = voltage_a
current[n_idx] = current_n; current[a_idx] = current_a
power[n_idx] = power_n;     power[a_idx] = power_a
pf[n_idx] = pf_n;           pf[a_idx] = pf_a
delta_i[n_idx] = delta_n;   delta_i[a_idx] = delta_a
anomaly_dur[n_idx] = anom_n; anomaly_dur[a_idx] = anom_a
labels[n_idx] = "Normal";    labels[a_idx] = "Abnormal"

#  Build DataFrame 
df = pd.DataFrame({
    "Timestamp": timestamps,
    "Voltage_V": np.round(voltage, 3),
    "Current_A": np.round(current, 3),
    "Power_W": np.round(power, 3),
    "Power_Factor": np.round(pf, 4),
    "Season": seasons,
    "Time_of_Day": times_of_day,
    "Delta_I": np.round(delta_i, 3),
    "Anomaly_Duration_min": anomaly_dur,
    "Label": labels
})

#  Verify overlap 
print("\nClass Distribution:")
print(df["Label"].value_counts())
print("\nFeature Stats (verify overlap):")
for col in ["Voltage_V","Current_A","Power_W","Power_Factor","Delta_I","Anomaly_Duration_min"]:
    n_v = df.loc[df["Label"]=="Normal", col]
    a_v = df.loc[df["Label"]=="Abnormal", col]
    print(f"  {col}:")
    print(f"    Normal:   mean={n_v.mean():.3f}, std={n_v.std():.3f}")
    print(f"    Abnormal: mean={a_v.mean():.3f}, std={a_v.std():.3f}")

df.to_csv(OUTPUT_CSV, index=False)
print(f"\n Dataset saved: {OUTPUT_CSV}")

#  Clear notebook outputs 
if os.path.exists(NOTEBOOK_PATH):
    print("Clearing notebook outputs...")
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Notebook outputs cleared.")

print("\n Expected accuracy: 75-88% (NOT 100%)")
print("   Re-run all cells in main.ipynb")
print("=" * 60)
