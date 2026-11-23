"""SuperKart sales forecasting UI (Streamlit). Calls the Flask backend for online and batch inference."""
import os

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7860").rstrip("/")
FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category"]

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="wide")
st.title("🛒 SuperKart Sales Forecaster")
st.caption("Forecast total revenue for a product in a store, one at a time or in bulk.")


def call_api(method_path, **kwargs):
    """POST to the backend and return (json, error_message)."""
    try:
        resp = requests.post(f"{BACKEND_URL}{method_path}", timeout=60, **kwargs)
        body = resp.json()
        if resp.status_code != 200:
            return None, body.get("error", f"HTTP {resp.status_code}")
        return body, None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the prediction service at {BACKEND_URL}. Is the backend container running?"
    except requests.exceptions.Timeout:
        return None, "The prediction service timed out. Please try again."
    except ValueError:
        return None, "The prediction service returned an invalid response."


tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

with tab_single:
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Product")
            product_id_char = st.selectbox("Product family (ID prefix)", ["FD", "DR", "NC"],
                                           help="FD = Food, DR = Drinks, NC = Non-consumables")
            product_type_category = st.selectbox("Product type category", ["Perishables", "Non Perishables"])
            sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        with c2:
            st.subheader("Product metrics")
            weight = st.number_input("Product weight", min_value=0.0, max_value=50.0, value=12.66, step=0.01)
            mrp = st.number_input("Product MRP", min_value=0.0, max_value=1000.0, value=117.08, step=0.01)
            area = st.number_input("Allocated display area (ratio 0-1)", min_value=0.0, max_value=1.0,
                                   value=0.027, step=0.001, format="%.3f")
        with c3:
            st.subheader("Store")
            store_type = st.selectbox("Store type", ["Supermarket Type2", "Supermarket Type1",
                                                     "Departmental Store", "Food Mart"])
            store_size = st.selectbox("Store size", ["Medium", "High", "Small"])
            city_tier = st.selectbox("City tier", ["Tier 2", "Tier 1", "Tier 3"])
            store_age = st.number_input("Store age (years)", min_value=0, max_value=100, value=16, step=1)
        submitted = st.form_submit_button("Predict sales", type="primary")

    if submitted:
        payload = {
            "Product_Weight": weight, "Product_Sugar_Content": sugar, "Product_Allocated_Area": area,
            "Product_MRP": mrp, "Store_Size": store_size, "Store_Location_City_Type": city_tier,
            "Store_Type": store_type, "Product_Id_char": product_id_char, "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_category,
        }
        with st.spinner("Forecasting..."):
            result, error = call_api("/v1/predict", json=payload)
        if error:
            st.error(error)
        else:
            st.metric("Forecast product-store sales", f"{result['Predicted_Sales']:,.2f}")
            for w in result.get("warnings", []):
                st.warning(w)
            with st.expander("Request sent to the API"):
                st.json(payload)

with tab_batch:
    st.write("Upload a CSV with these columns: " + ", ".join(f"`{c}`" for c in FEATURES))
    uploaded = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception:
            st.error("Could not read the file as CSV.")
            st.stop()
        missing = [c for c in FEATURES if c not in batch_df.columns]
        if missing:
            st.error(f"Missing column(s): {missing}")
            st.stop()
        st.write(f"Preview ({len(batch_df)} rows):")
        st.dataframe(batch_df.head(), use_container_width=True)
        if st.button("Predict batch", type="primary"):
            with st.spinner("Scoring batch..."):
                result, error = call_api("/v1/predictbatch",
                                         files={"file": ("batch.csv", batch_df.to_csv(index=False).encode("utf-8"), "text/csv")})
            if error:
                st.error(error)
            else:
                out = batch_df.copy()
                out["Predicted_Sales"] = [result[str(i)] for i in range(len(out))]
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows scored", len(out))
                m2.metric("Total forecast", f"{out['Predicted_Sales'].sum():,.2f}")
                m3.metric("Average forecast", f"{out['Predicted_Sales'].mean():,.2f}")
                st.dataframe(out, use_container_width=True)
                st.download_button("Download predictions as CSV", out.to_csv(index=False).encode("utf-8"),
                                   file_name="superkart_batch_predictions.csv", mime="text/csv")
