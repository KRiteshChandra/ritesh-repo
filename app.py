import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load models and encoder
reg_model = joblib.load("linear_model.pkl")
clf_model = joblib.load("rf_classifier.pkl")
le = joblib.load("label_encoder.pkl")

CO2_FACTOR = 0.475        # kg CO2 per kWh
TARIFF_DAY, TARIFF_NIGHT = 0.15, 0.09   # Example tariff rates

st.title("⚡ Smart Energy Forecast Assistant")
st.markdown("Easily predict your appliances' **future energy use, electricity costs, and CO₂ emissions** — plus get smart recommendations for when to run them.")

# User input: list of appliances from the dataset
all_devices = list(le.classes_)
selected_devices = st.multiselect("Select Appliances to forecast:", all_devices, default=[all_devices[0]])

# Forecast settings
days = st.slider("Days to Forecast:", 3, 14, 7)
duration = st.slider("Average Usage Hours per Day:", 1, 12, 3)
start_date = st.date_input("Start Date", pd.to_datetime("2022-09-24"))

if st.button("Run Forecast"):
    hours_ahead = 24 * days
    future = pd.date_range(start=start_date, periods=hours_ahead, freq="h")
    results = []
    
    # Forecast for each selected appliance
    for device in selected_devices:
        dev_id = le.transform([device])[0]
        df = pd.DataFrame({
            "hour": future.hour,
            "day": future.day,
            "weekday": future.weekday,
            "node_encoded": dev_id,
            "timestamp": future
        })
        watts = reg_model.predict(df.drop(columns=["timestamp"]))
        df["Watts"] = watts
        df["Energy_kWh"] = (watts * duration) / 1000
        df["CO2_kg"] = df["Energy_kWh"] * CO2_FACTOR
        df["Cost_$"] = df.apply(
            lambda r: r["Energy_kWh"] * (TARIFF_NIGHT if (r["hour"] >= 22 or r["hour"] < 6) else TARIFF_DAY),
            axis=1
        )
        df["Appliance"] = device
        results.append(df)

    final = pd.concat(results)

    # Appliance-level summary
    summary = final.groupby("Appliance")[["Energy_kWh","CO2_kg","Cost_$"]].sum()
    
    # Show results
    st.write("### 📊 Appliance Forecast Summary")
    st.dataframe(summary.style.format({"Energy_kWh": "{:.2f}", "CO2_kg": "{:.2f}", "Cost_$": "${:.2f}"}))

    total_energy = summary["Energy_kWh"].sum()
    total_cost = summary["Cost_$"].sum()
    total_co2 = summary["CO2_kg"].sum()

    st.write(f"**Total across all appliances for {days} days:**")
    st.write(f"- 🔋 Energy: {total_energy:.1f} kWh")
    st.write(f"- 💵 Cost: ${total_cost:.2f}")
    st.write(f"- 🌍 Emissions: {total_co2:.1f} kg CO₂")
    st.write(f"That’s like driving ~{total_co2*4:.0f} km in a petrol car 🚗.")

    # ========= Recommendations =========
    st.subheader("💡 Smart Recommendations")

    # Best / Worst day across all selected devices
    daily_summary = final.groupby(final["timestamp"].dt.date)[["Energy_kWh","CO2_kg","Cost_$"]].sum()
    best_day = daily_summary["Cost_$"].idxmin()
    worst_day = daily_summary["Cost_$"].idxmax()

    st.success(f"✅ Best day to run energy-heavy appliances: **{best_day}** "
               f"(~${daily_summary.loc[best_day,'Cost_$']:.2f})")
    st.error(f"⚠️ Avoid heavy use on: **{worst_day}** "
             f"(~${daily_summary.loc[worst_day,'Cost_$']:.2f})")

    # Top contributors
    top_cost_appliance = summary["Cost_$"].idxmax()
    top_co2_appliance = summary["CO2_kg"].idxmax()
    st.warning(f"💰 Biggest Cost Driver: **{top_cost_appliance}** (~${summary['Cost_$'].max():.2f})")
    st.warning(f"🌍 Highest Carbon Impact: **{top_co2_appliance}** (~{summary['CO2_kg'].max():.1f} kg CO₂)")

    # ========= Visualization =========
    st.subheader("📊 Cost Share by Appliance")
    fig, ax = plt.subplots()
    ax.pie(summary["Cost_$"], labels=summary.index, autopct='%1.1f%%')
    st.pyplot(fig)

    st.subheader("🌍 Carbon Emissions per Appliance")
    st.bar_chart(summary["CO2_kg"])

    st.subheader("💵 Daily System Cost Forecast")
    st.line_chart(daily_summary["Cost_$"])
