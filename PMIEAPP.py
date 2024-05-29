import pandas as pd
import joblib
import streamlit as st

# Load the saved models
models = {}
for model_name in ["DecisionTree", "GradientBoosting", "RandomForest", "XGBoost"]:
  try:
    models[model_name] = joblib.load(f"c:/Users/maluser/Desktop/models/{model_name}.pkl")
  except FileNotFoundError:
    st.error(f"Model '{model_name}' not found. Please ensure it's saved in the 'saved_models' directory.")

# Load the DataFrame (assuming it's saved as 'predictive_maintenance.csv')
df = pd.read_csv('c:/Users/maluser/Downloads/predictive_maintenance.csv')

# Feature names for user input
features = df.columns[2:9]  # Features from index 2 to 8 (excluding UDI and Product ID)

def predict_failure(model, data):
  """Predicts failure using the specified model and data."""
  return model.predict_proba(data)[0][1]  # Probability of failure (class 1)

def main():
  """Streamlit app for predictive maintenance."""

  # Title and introduction
  st.title("Predictive Maintenance System")
  st.write("This system allows you to input machine parameters and predict potential failures using various machine learning models.")

  # User input for parameters
  user_input = {}
  for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature} value:")

  # Prediction buttons for each model
  model_predictions = {}
  for model_name, model in models.items():
    if st.button(f"Predict with {model_name}"):
      data = pd.DataFrame([user_input]).values  # Convert user input to DataFrame
      probability = predict_failure(model, data)
      model_predictions[model_name] = probability

  # Display predictions
  if model_predictions:
    st.subheader("Prediction Results:")
    for model_name, probability in model_predictions.items():
      failure_risk = "High" if probability > 0.5 else "Low"
      st.write(f"{model_name}: Probability of failure: {probability:.2f} ({failure_risk})")

if __name__ == "__main__":
  main()
