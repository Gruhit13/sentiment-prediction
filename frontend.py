import streamlit as st
import requests
import json

st.title('Sentiment Prediction')
st.subheader('Eager to know attitude of some text. You came to right place.')

# Take image input
text = st.text_input("text", placeholder="I am feeling positive today")

CLASS_TO_INDEX = {
    'Positive': 0, 
    'Neutral': 1, 
    'Negative': 2
}

def predict():
    URL = "https://gruhit-patel-sentimentanalysis.hf.space/predict"

    if len(text) != 0:
        with st.spinner("Waiting for model response..."):
            resp = requests.post(
                URL,
                json={"text": text}
            )

            if hasattr(resp, "_content"):
                result = json.loads(resp._content.decode('utf-8'))
                result = json.loads(result['result'])

                label = CLASS_TO_INDEX[result['sentiment']]
                probs = result['probs'][label]*100

                template = "#### Sentiment of text is :{color}[{sentiment}] with {confidence:.2f}% confidence"

                if result['sentiment'] == "Positive":
                    template = template.format(color='green', sentiment="Positive", confidence=probs)
                elif result['sentiment'] == "Negative":
                    template = template.format(color='red', sentiment="Negative", confidence=probs)
                else:
                    template = template.format(color='blue', sentiment="Neutral", confidence=probs)
                
                st.markdown(template)

st.button(
    "Predict",
    on_click=predict,
    type="primary"
)