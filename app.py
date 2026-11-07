import streamlit as st
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="🤖",
    layout="centered"
)

# 2. Define Classifier class so pickle can recognize it in __main__
class SimpleClassifier:
    def predict(self, X):
        predictions = []
        for sample in X:
            age, spend, calls, tenure = sample
            if calls >= 3 or (spend < 200 and tenure < 6):
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    def predict_proba(self, X):
        probs = []
        for sample in X:
            calls = sample[2]
            if calls >= 3:
                probs.append([0.15, 0.85])
            else:
                probs.append([0.90, 0.10])
        return probs

# 3. Load the trained ML Model
@st.cache_resource
def load_model():
    with open('churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 4. App Header
st.title("🤖 Customer Churn Prediction ML App")
st.markdown("This machine learning web application predicts whether a customer is likely to stay or leave based on their behavior patterns.")
st.markdown("---")

# 5. Sidebar Inputs for User
st.sidebar.header("📊 Customer Profile Inputs")

age = st.sidebar.slider("Customer Age", 18, 70, 30)
monthly_spend = st.sidebar.number_input("Monthly Spend ($)", 50, 1000, 250)
support_calls = st.sidebar.slider("Customer Support Calls", 0, 10, 1)
tenure_months = st.sidebar.slider("Tenure (Months)", 1, 72, 12)

# 6. Prediction Logic
if st.sidebar.button("Predict Churn Status", type="primary"):
    input_data = [[age, monthly_spend, support_calls, tenure_months]]
    
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("🔍 Prediction Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk:** This customer is **Likely to Churn (Leave)**! (Confidence: {prediction_proba[0][1]*100:.2f}%)")
        st.markdown("*Recommendation:* Offer a special loyalty discount or assign a support representative immediately.")
    else:
        st.success(f"✅ **Low Risk:** This customer is **Likely to Stay**! (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        st.markdown("*Recommendation:* Keep up the great service experience!")

st.markdown("---")
st.markdown("### 💡 About This ML Project")
st.markdown("Built using **Python and Streamlit**. Perfect for Data Science & ML Engineering portfolios.")