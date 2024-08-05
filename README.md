# sentiment-prediction

For identifying the sentiment in the text I used the current state-of-art models which are currently ruling AI market, transformers.
For this task I made use of just encoder only model. The model is a finetuned version of a basemodel that intially ended up in overfitting.
Overfiting is normal in transformer model as their parameters raises exponentially. 

Along with a model I also have implemented a text preprocessor pipeline the following things before making a prediction
1. lower the text
2. replace all hastags(#)
3. remove any mention tags(@)
4. remove any hyperlink
5. remove any leading and trailing spaces
6. convert emoji to their corresponding name
7. lemmatize the word to remove unnecessary information
8. remove stopwords from the sentence
9. convert the sentence words to tokens

### Frontend
Have a look at the application by clicking the link 👉: [Sentiment Predictor](https://sentiment-predictor.streamlit.app/)

### Backend
For deployment of backend server we have made use of HuggingFace 👉: [Backend](https://huggingface.co/spaces/gruhit-patel/SentimentAnalysis/tree/main) <br>
Due to github's limitation on file size you can clone the hugging face repo to replicate the backend 

#### App Demo
[streamlit-frontend-2024-08-05-18-08-50.webm](https://github.com/user-attachments/assets/aa90292f-73e5-4c53-97e8-8792432961ab)
