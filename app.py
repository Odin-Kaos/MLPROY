import gradio as gr
import requests

API_URL = "https://lab3-api-latest.onrender.com"

FEATURES = [
    "Breathing Problem",
    "Fever",
    "Dry Cough",
    "Sore Throat",
    "Running Nose",
    "Asthma",
    "Chronic Lung Disease",
    "Headache",
    "Heart Disease",
    "Diabetes",
    "Hyper Tension",
    "Fatigue",
    "Gastrointestinal",
    "Abroad Travel",
    "Contact with COVID Patient",
    "Attended Large Gathering",
    "Visited Public Exposed Places",
    "Family Working in Public Exposed Places",
    "Wearing Masks",
    "Sanitization from Market",
]

def predict_from_checkboxes(selected):
    # Convert selected labels → binary vector
    vector = ["1" if f in selected else "0" for f in FEATURES]
    vect_str = ",".join(vector)

    try:
        response = requests.post(
            f"{API_URL}/predict",
            data={"vect": vect_str},
            timeout=20
        )
        response.raise_for_status()
        return {
            "vector_sent": vect_str,
            "api_response": response.json()
        }
    except Exception as e:
        return {"error": str(e)}

iface = gr.Interface(
    fn=predict_from_checkboxes,
    inputs=gr.CheckboxGroup(FEATURES, label="Select Features"),
    outputs=gr.JSON(),
    title="Symptom-Based Classifier",
    description="Select symptoms or conditions. The app converts them into a binary vector and sends it to the API."
)

if __name__ == "__main__":
    iface.launch()
