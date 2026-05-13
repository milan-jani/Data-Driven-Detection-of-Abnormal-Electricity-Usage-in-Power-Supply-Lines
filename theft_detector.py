import joblib
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_system():
    ada = joblib.load('models/adaboost_model.joblib')
    rf = joblib.load('models/random_forest_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return ada, rf, scaler

# Models ko memory mein rakhne ke liye global load
try:
    ada_model, rf_model, scaler = load_system()
except:
    pass

def check_my_data(voltage, current, power, pf, season, time_of_day, delta_i, duration=0):
    """
    User input lekar prediction dene wala function.
    Season: 0=Winter, 1=Summer
    Time: 0=Afternoon, 1=Evening, 2=Morning, 3=Night
    """
    data = [voltage, current, power, pf, season, time_of_day, delta_i, duration]
    columns = ['Voltage_V', 'Current_A', 'Power_W', 'Power_Factor', 'Season', 'Time_of_Day', 'Delta_I', 'Anomaly_Duration_min']
    
    df_input = pd.DataFrame([data], columns=columns)
    data_scaled = scaler.transform(df_input)
    
    print(f"\n--- Prediction for V:{voltage}V, I:{current}A ---")
    
    # AdaBoost Result
    ada_pred = ada_model.predict(data_scaled)[0]
    ada_prob = ada_model.predict_proba(data_scaled)[0][1]
    res_ada = "THEFT" if ada_pred == 1 else "NORMAL"
    print(f"AdaBoost Result     : {res_ada:10} (Prob: {ada_prob:.2%})")
    
    # Random Forest Result
    rf_pred = rf_model.predict(data_scaled)[0]
    rf_prob = rf_model.predict_proba(data_scaled)[0][1]
    res_rf = "THEFT" if rf_pred == 1 else "NORMAL"
    print(f"Random Forest Result: {res_rf:10} (Prob: {rf_prob:.2%})")

if __name__ == "__main__":
    # Example usage:
    print("Manual Check Example:")
    check_my_data(230, 0.4, 92, 0.92, 1, 0, 0.01) # Normal
    check_my_data(210, 2.0, 420, 0.70, 0, 3, 0.88) # Theft
    print("stable parameters (normal load)")
    check_my_data(228, 0.5, 100, 0.95, 1, 2, 0.02, 0)

    # 2. Morning moderate usage (fridge + lights)
    check_my_data(232, 1.2, 260, 0.90, 1, 2, 0.05, 5)

    # 3. Afternoon minimal load
    check_my_data(235, 0.3, 70, 0.97, 1, 0, 0.01, 0)

    # 4. Evening peak (TV + fan + lights)
    check_my_data(225, 2.5, 520, 0.92, 0, 1, 0.08, 10)

    # 5. Night stable load (AC running normally)
    check_my_data(230, 4.5, 950, 0.88, 1, 3, 0.07, 20)

    print("unstable parameters (theft load)")
    # 6. Sudden current spike but PF drops (illegal tapping)
    check_my_data(215, 6.0, 800, 0.60, 1, 1, 0.75, 15)

    # 7. High current but power mismatch (bypass meter scenario)
    check_my_data(220, 8.0, 900, 0.50, 0, 3, 0.90, 25)

    # 8. Large delta current fluctuation
    check_my_data(210, 5.5, 1000, 0.70, 0, 1, 0.65, 10)

    # 9. Abnormal current at low voltage (line tapping load)
    check_my_data(205, 7.0, 850, 0.55, 1, 3, 0.85, 30)

    # 10. Continuous anomaly (steady theft)
    check_my_data(218, 4.8, 700, 0.65, 0, 0, 0.60, 60)
