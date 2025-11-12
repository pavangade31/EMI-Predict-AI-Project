import streamlit as st
import pandas as pd
import pickle

# -------------------- Load Models --------------------
with open("EMI Regression Model.pkl", "rb") as f:
    regression_model = pickle.load(f)

with open("EMI Classifier Model.pkl", "rb") as f:
    classification_model = pickle.load(f)

# -------------------- Define Input Columns --------------------
categorical_cols = {
    'Gender': ['Male', 'Female', 'Other'],
    'Marital Status': ['Married', 'Single'],
    'Education': ['High School', 'Graduate', 'Post Graduate', 'Professional'],
    'Employment Type': ['Private', 'Self-employed', 'Government', 'Business'],
    'Company Type': ['MNC', 'Mid-size', 'Small', 'Startup'],
    'House Type': ['Own', 'Rented'],
    'Emi Scenario': ['Education EMI', 'Home Appliances EMI', 'Personal Loan EMI', 'Vehicle EMI'],
    'Existing Loans': ['Yes', 'No'],
    'Emi Eligibility': ['Eligible', 'Not_Eligible', 'High_Risk']
}

numerical_cols = [
    'Age', 'Monthly Salary', 'Years Of Employment', 'Monthly Rent',
    'Family Size', 'Dependents', 'School Fees', 'College Fees',
    'Travel Expenses', 'Groceries Utilities', 'Other Monthly Expenses',
    'Current Emi Amount', 'Credit Score', 'Bank Balance', 'Emergency Fund',
    'Requested Amount', 'Requested Tenure', 'Max Monthly Emi'
]

# -------------------- Define Expected Columns --------------------
# Regression model expects 33 columns
regression_columns = [
    'Age', 'Monthly Salary', 'Years Of Employment', 'Monthly Rent',
    'Family Size', 'Dependents', 'School Fees', 'College Fees',
    'Travel Expenses', 'Groceries Utilities', 'Other Monthly Expenses',
    'Current Emi Amount', 'Credit Score', 'Bank Balance', 'Emergency Fund',
    'Requested Amount', 'Requested Tenure', 'Max Monthly Emi',
    'Gender_Male', 'Marital Status_Single', 'Education_High School',
    'Education_Post Graduate', 'Education_Professional',
    'Employment Type_Private', 'Employment Type_Self-employed',
    'Company Type_MNC', 'Company Type_Mid-size', 'Company Type_Small',
    'Company Type_Startup', 'House Type_Own', 'House Type_Rented',
    'Emi Scenario_Education EMI', 'Emi Scenario_Home Appliances EMI',
    'Emi Scenario_Personal Loan EMI', 'Emi Scenario_Vehicle EMI',
    'Existing Loans_Yes'
][:33]

# Classification model expects 37 columns
classification_columns = [
    'Age', 'Monthly Salary', 'Years Of Employment', 'Monthly Rent',
    'Family Size', 'Dependents', 'School Fees', 'College Fees',
    'Travel Expenses', 'Groceries Utilities', 'Other Monthly Expenses',
    'Current Emi Amount', 'Credit Score', 'Bank Balance', 'Emergency Fund',
    'Requested Amount', 'Requested Tenure', 'Max Monthly Emi',
    'Gender_Male', 'Marital Status_Single', 'Education_High School',
    'Education_Post Graduate', 'Education_Professional',
    'Employment Type_Private', 'Employment Type_Self-employed',
    'Company Type_MNC', 'Company Type_Mid-size', 'Company Type_Small',
    'Company Type_Startup', 'House Type_Own', 'House Type_Rented',
    'Emi Scenario_Education EMI', 'Emi Scenario_Home Appliances EMI',
    'Emi Scenario_Personal Loan EMI', 'Emi Scenario_Vehicle EMI',
    'Existing Loans_Yes', 'Emi Eligibility_High_Risk',
    'Emi Eligibility_Not_Eligible'
][:37]

# -------------------- Utility Function --------------------
def preprocess_input(df, feature_list):
    """One-hot encode, fix missing/extra columns, and align with training features."""
    df_encoded = pd.get_dummies(df, columns=categorical_cols.keys(), drop_first=True)

    # Remove unexpected columns
    extra_cols = [c for c in df_encoded.columns if c not in feature_list]
    if extra_cols:
        df_encoded.drop(columns=extra_cols, inplace=True)

    # Add missing columns as 0
    for col in feature_list:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder to match model
    df_encoded = df_encoded[feature_list]
    return df_encoded

# -------------------- App Configuration --------------------
st.set_page_config(page_title="EMI Prediction Suite", page_icon="💰", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4B0082;'>💰 EMI Prediction & Eligibility App</h1>", unsafe_allow_html=True)
st.write("Choose between **EMI Amount Prediction** and **Eligibility Prediction** below:")

# -------------------- Sidebar Selection --------------------
option = st.sidebar.radio(
    "Select Prediction Type:",
    ("📆 EMI Amount Prediction", "⭐ EMI Eligibility Prediction")
)

# -------------------- Collect Inputs --------------------
user_input = {}

st.subheader("Enter Your Details 👇")

for col, options in categorical_cols.items():
    user_input[col] = st.selectbox(f"{col}", options)

for col in numerical_cols:
    user_input[col] = st.number_input(f"{col}", min_value=0, value=0)

input_df = pd.DataFrame([user_input])

# -------------------- EMI Amount Prediction --------------------
if option == "📆 EMI Amount Prediction":
    st.markdown("### 🪙 Predict Your Monthly EMI Amount")

    if st.button("🔮 Predict EMI Amount"):
        try:
            input_encoded = preprocess_input(input_df, regression_columns)
            prediction = regression_model.predict(input_encoded)[0]
            st.success(f"✅ Predicted EMI Amount: **₹{prediction:.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
    else:
        st.info("Enter details and click Predict to see EMI Amount.")

# -------------------- EMI Eligibility Prediction --------------------
elif option == "⭐ EMI Eligibility Prediction":
    st.markdown("### 💳 Check Your EMI Eligibility")

    if st.button("📈 Predict Eligibility"):
        try:
            input_encoded = preprocess_input(input_df, classification_columns)
            prediction = classification_model.predict(input_encoded)[0]
            eligibility_map = {0: "High Risk", 1: "Not Eligible", 2: "Eligible"}
            label = eligibility_map.get(int(prediction), f"Unknown ({prediction})")
            st.success(f"🎯 EMI Eligibility Result: **{label}**")
        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")
    else:
        st.info("Enter details and click Predict to check eligibility.")

# -------------------- View Input Data --------------------
with st.expander("📄 View Input Data"):
    st.dataframe(input_df)
