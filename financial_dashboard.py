import streamlit as st
import pandas as pd
import requests
import json
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#  API setup
API_KEY = "d75afeaf-383e-456c-9814-bd84f8c39041"
url = "https://api.financialdatasets.ai/v1/income_statements"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/json"
}
params = {
    "ticker": "AAPL",
    "period": "annual",
    "limit": 10
}

# Try API request, fallback to local or generated JSON
try:
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()["income_statements"]
except Exception as e:
    st.warning(f" API failed: {e}")
    try:
        with open("fallback_income_statements.json", "r") as f:
            data = json.load(f)["income_statements"]
    except FileNotFoundError:
        st.info("Generating fallback data...")
        data = [
            {"calendar_date": "2023-12-31", "earnings_per_share": 6.16, "revenue": 383285000000, "net_income": 96995000000},
            {"calendar_date": "2022-12-31", "earnings_per_share": 6.15, "revenue": 394328000000, "net_income": 99803000000},
            {"calendar_date": "2021-12-31", "earnings_per_share": 5.67, "revenue": 365817000000, "net_income": 94680000000},
            {"calendar_date": "2020-12-31", "earnings_per_share": 3.31, "revenue": 274515000000, "net_income": 57411000000},
            {"calendar_date": "2019-12-31", "earnings_per_share": 2.99, "revenue": 260174000000, "net_income": 55256000000}
        ]

# Build DataFrame
df = pd.DataFrame([{
    "year": int(item["calendar_date"][:4]),
    "eps": item["earnings_per_share"],
    "revenue": item["revenue"],
    "net_income": item["net_income"]
} for item in data])
df = df.sort_values("year").reset_index(drop=True)

#  Train EPS model
model = LinearRegression().fit(df[['year']], df['eps'])

#  Sidebar controls
st.sidebar.header("🔧 Controls")
selected_year = st.sidebar.slider("Select year to forecast EPS", 2024, 2030)
show_all_predictions = st.sidebar.checkbox("Show EPS forecast for 2024–2030", value=True)
show_revenue_chart = st.sidebar.checkbox("Show Revenue Chart", value=True)
show_net_income_chart = st.sidebar.checkbox("Show Net Income Chart", value=True)

#  Single year prediction
future_df = pd.DataFrame({'year': [selected_year]})
predicted_eps = model.predict(future_df)[0]
st.write(f" Predicted EPS for {selected_year}: **${predicted_eps:.2f}**")

#  Forecast multiple years
if show_all_predictions:
    future_years = pd.DataFrame({'year': list(range(2024, 2031))})
    future_eps = model.predict(future_years)
    forecast_df = pd.DataFrame({
        'year': future_years['year'],
        'predicted_eps': future_eps
    })
    st.subheader("EPS Forecast (2024–2030)")
    st.line_chart(forecast_df.set_index('year'))

#  Combined EPS chart
combined_df = pd.concat([
    df.assign(type='Actual'),
    forecast_df.assign(type='Predicted')
])
fig, ax = plt.subplots()
for label, group in combined_df.groupby('type'):
    ax.plot(group['year'], group['eps' if label == 'Actual' else 'predicted_eps'],
            marker='o', label=label)
ax.set_title("EPS: Actual vs Predicted")
ax.set_xlabel("Year")
ax.set_ylabel("EPS ($)")
ax.legend()
st.pyplot(fig)

#  Revenue chart
if show_revenue_chart:
    st.subheader(" Revenue Trend")
    st.line_chart(df.set_index("year")["revenue"])

#  Net income chart
if show_net_income_chart:
    st.subheader(" Net Income Trend")
    st.line_chart(df.set_index("year")["net_income"])
