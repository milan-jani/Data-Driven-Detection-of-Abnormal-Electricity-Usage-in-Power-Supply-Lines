# ⚡ Electricity Theft Detection Using Machine Learning

---

## 1. Problem Statement
Electricity theft is a major issue in rural and urban India. People illegally tap distribution lines (pole wires) bypassing the meter, causing financial losses to electricity boards and honest consumers. The goal is to build an ML-based system that detects abnormal electricity consumption patterns automatically.

## 2. Dataset Description
File: `electricity_theft_dataset.csv`
Total Rows: 5000
Type: Synthetic — generated based on realistic electrical behavior

### Columns:
| Column | Type | Description |
|---|---|---|
| Timestamp | String | Date & time of reading (every 30 min) |
| Voltage_V | Float | Supply voltage in Volts (normal: 215-225V) |
| Current_A | Float | Current drawn in Amperes (normal: 0.4-0.9A) |
| Power_W | Float | Power consumption in Watts (V × I) |
| Power_Factor| Float | Electrical efficiency (normal: 0.88-0.96) |
| Season | String | Winter / Summer |
| Time_of_Day | String | Morning / Afternoon / Evening / Night |
| Delta_I | Float | Rate of change of current (sudden spike indicator) |
| Anomaly_Duration_min | Int | How long abnormal condition lasted (0 if normal) |
| Label | String | Normal / Abnormal (target variable) |

### Class Distribution:
*   Normal: 4283 rows (85%)
*   Abnormal: 717 rows (15%)
*   Note: 75% of abnormal cases occur at Night.

## 3. Parameter Comparison
| Parameter | Normal | Abnormal (Theft) |
|---|---|---|
| Current | 0.4 — 0.9A | 1.5 — 3.8A (sudden jump) |
| Voltage | 215 — 225V | Drops to 197 — 210V |
| Power | 90 — 200W | 350 — 800W |
| Power Factor | 0.88 — 0.96 | Drops to 0.60 — 0.78 |
| Delta_I | 0.01 — 0.05 | 0.8 — 2.5 (sudden spike) |
| Time | Any | Mostly Night (11PM — 5AM) |

## 4. Why ML?
*   **Handles High Load:** Can distinguish between summer afternoon load vs. theft.
*   **Pattern Detection:** Detects gradual theft that thresholding might miss.
*   **Adaptive:** Learns seasonal and time-based variations.
