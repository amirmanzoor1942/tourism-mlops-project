"""Streamlit frontend for tourism package purchase prediction."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Tourism Package Purchase Prediction")
st.write(
    "Enter customer details to estimate purchase propensity "
    "and help prioritize marketing outreach."
)

if not MODEL_PATH.is_file():
    st.error("The trained model is unavailable. Complete model training first.")
    st.stop()

model = load_model()

with st.form("customer_details"):
    left, right = st.columns(2)

    with left:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35
        )
        city_tier = st.selectbox("City tier", [1, 2, 3])
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Free Lancer"],
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        persons = st.number_input(
            "Number of people visiting",
            min_value=1, max_value=20, value=3,
        )
        property_star = st.selectbox(
            "Preferred hotel star rating", [3, 4, 5]
        )
        marital_status = st.selectbox(
            "Marital status",
            ["Married", "Divorced", "Unmarried", "Single"],
        )

    with right:
        trips = st.number_input(
            "Number of trips per year",
            min_value=0, max_value=100, value=3,
        )
        passport = st.selectbox("Valid passport", ["No", "Yes"])
        own_car = st.selectbox("Owns a car", ["No", "Yes"])
        children = st.number_input(
            "Number of children visiting",
            min_value=0, max_value=20, value=1,
        )
        designation = st.selectbox(
            "Job designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        )
        income = st.number_input(
            "Gross monthly income (dataset currency units)",
            min_value=0.0,
            value=23000.0,
            step=500.0,
        )

    submitted = st.form_submit_button("Predict purchase propensity")

if submitted:
    if children > persons:
        st.error(
            "The number of children cannot exceed the total "
            "number of people visiting."
        )
    else:
        customer = pd.DataFrame([{
            "Age": age,
            "CityTier": city_tier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": trips,
            "Passport": int(passport == "Yes"),
            "OwnCar": int(own_car == "Yes"),
            "NumberOfChildrenVisiting": children,
            "Designation": designation,
            "MonthlyIncome": income,
        }])

        # Match the training column order.
        customer = customer.loc[:, model.feature_names_in_]

        positive_index = list(model.classes_).index(1)
        score = float(model.predict_proba(customer)[0, positive_index])
        prediction = int(model.predict(customer)[0])

        st.metric("Estimated purchase propensity", f"{score:.1%}")

        if prediction == 1:
            st.success("Predicted purchaser — prioritize for outreach.")
        else:
            st.info("Predicted non-purchaser at the current threshold.")

        st.caption(
            "Classification threshold: 50%. The score is a model estimate, "
            "not a guaranteed or calibrated purchase probability."
        )

        with st.expander("View customer inputs"):
            st.dataframe(customer, hide_index=True)

st.caption(
    "This model uses customer profile information and excludes "
    "sales-pitch and follow-up data."
)
