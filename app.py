from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import contractions
import re
import json
import plotly.graph_objs as go
import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertConfig
import spacy
import numpy as np
import tensorflow as tf
import tensorflow.lite as lite
import os
import base64

nlp = spacy.load('en_core_web_sm')

stopwords = set(nltk.corpus.stopwords.words('english'))

@st.cache_resource()
def normalize_document1(doc):
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    # word lemmatization
    doc = nlp(doc)
    tokens = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in doc]
    #tokens = [wnl.lemmatize(token) for token in tokens if not token.isnumeric()]
    # removing any single character words \ numbers \ symbols
    tokens = [token for token in tokens if len(token) > 1] 
    #filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stopwords]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

@st.cache_resource()
def normalize_corpus1(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        norm_doc = normalize_document1(doc)
        norm_docs.append(norm_doc)

    return norm_docs

@st.cache_resource()
def normalize_document2(doc):
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = contractions.fix(doc)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    return doc

@st.cache_resource()
def normalize_corpus2(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        norm_doc = normalize_document2(doc)
        norm_docs.append(norm_doc)

    return norm_docs

# Load the models and vectorizer
lr = joblib.load("data/models/lr.joblib")
cv = joblib.load("data/models/cv.joblib")
interpreter = lite.Interpreter(model_path="data/models/distilbert_model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@st.cache_resource()
def predict_label_model1(text):
    normalized_text = normalize_document1(text)
    X = cv.transform([normalized_text])
    prob_scores = lr.predict_proba(X)[0]
    top3_indices = prob_scores.argsort()[::-1][:3]
    top3_labels = lr.classes_[top3_indices]
    top3_probs = prob_scores[top3_indices] * 100

    #st.write("Top 3 predicted labels and their corresponding probabilities:")
    #for i in range(3):
    #    st.write(f"{i+1}. {top3_labels[i]}: {top3_probs[i]:.2f}%")
    
    return top3_labels, top3_probs, normalized_text

# Load the tokenizer for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

MAX_SEQ_LENGTH = 85

label_encoder = LabelEncoder()
label_encoder.fit(lr.classes_)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

@st.cache_resource()
def create_distilbert_input_features(docs, max_seq_length):
    all_ids, all_masks = [], []
    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):

        tokens = tokenizer.tokenize(doc)

        if len(tokens) > max_seq_length-2:
            tokens = tokens[0 : (max_seq_length-2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)

        # Zero-pad up to the sequence length.
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)

        all_ids.append(ids)
        all_masks.append(masks)

    encoded = np.array([all_ids, all_masks])

    return encoded

@st.cache_resource()
def predict_label_model2(text):
    # Normalize and tokenize the input sentence
    norm_sentence = normalize_document2(text)
    # tokens = tokenizer.tokenize(norm_sentence)
    
    # Convert tokens to input features
    input_ids, input_masks = create_distilbert_input_features([norm_sentence], max_seq_length=MAX_SEQ_LENGTH)
    input_ids = np.array(input_ids, dtype=np.int32)
    input_masks = np.array(input_masks, dtype=np.int32)
    
    # Set the input data
    interpreter.set_tensor(input_details[0]['index'], input_ids)
    interpreter.set_tensor(input_details[1]['index'], input_masks)

    # Invoke the interpreter
    interpreter.invoke()

    # Get the model prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class label
    predicted_label = np.argmax(prediction, axis=1)
    
    # Decode the predicted label back to the string label
    decoded_predicted_label = label_encoder.inverse_transform(predicted_label)

    # Get the top 3 most probable labels and their probabilities
    top_3_labels = np.argsort(prediction[0])[-3:][::-1]
    label_probs = prediction[0][top_3_labels] * 100

    #st.write("Top 3 predicted labels and their corresponding probabilities:")
    decoded_top_3_labels = label_encoder.inverse_transform(top_3_labels)
    #for i in range(len(decoded_top_3_labels)):
    #    st.write(f"{i+1}. {decoded_top_3_labels[i]}: {label_probs[i]:.2f}%")
    
    return decoded_top_3_labels, label_probs, norm_sentence


def plot_probabilities(labels, probabilities, title):
    # Reverse the order of labels and probabilities
    reversed_labels = labels[::-1]
    reversed_probabilities = probabilities[::-1]
    # Round the probabilities to two decimal places and make them bold
    text = [f"<b>{round(prob, 2)}%</b>" for prob in reversed_probabilities]

    trace = go.Bar(
        y=reversed_labels,
        x=reversed_probabilities,
        orientation='h',
        text=text,
        textposition='outside',
        insidetextfont=dict(size=14, color="white"),
        marker=dict(line=dict(color='rgba(0,0,0,1)', width=2))
    )

    # Add annotations for bold labels
    annotations = []
    for i, label in enumerate(reversed_labels):
        annotations.append(dict(xref='paper', yref='y', y=label, x=-0.2, text=f"<b>{label}</b>", showarrow=False, font=dict(size=14)))

    layout = go.Layout(
        title=title,
        xaxis=dict(title='Probability (%)', range=[0, max(reversed_probabilities) * 1.1]),
        yaxis=dict(title='Labels', showticklabels=False),
        annotations=annotations,
        margin=dict(l=150, r=150)
    )

    fig = go.Figure(data=[trace], layout=layout)
    st.plotly_chart(fig)

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data()
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


####################################### LAYOUT ####################################################
st.set_page_config(
    page_title="NASCAR com classificator",
    page_icon="🏁",
)

st.title("NASCAR Radio communication label classification")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Intro', 'The process', 'Labels info', 'Try yourself', 'About us'])

with tab1:
    with st.container():
        # App header
        header_image = Image.open("data/images/hi-res-156047552_crop_exact.jpg")
        st.image(header_image, use_column_width=True)
    with st.container():
        # App title
        st.write("Welcome to our label classification app!")
        st.write("This app uses a prediction model to classify short text/sentences into labels.")
        st.write("Simply enter a sentence and the app will output the corresponding label.")

with tab2:
    with st.container():
        # Database screenshot
        database_screenshot = Image.open("data/images/database_screenshot.png")
        st.image(database_screenshot, use_column_width=True)

with tab3:
    with st.container():
        st.write("hey")


with tab4:
    with st.container():
        # Model selector
        model_options = ["Select a model", "Classical Machine Learning for NLP", "Transformer Machine Learning for NLP"]
        header_image = Image.open("data/images/comics.png")
        st.image(header_image, use_column_width=True)
        st.write("This is a web app that uses two different machine learning models to classify short texts/sentences of __max 85 words__ into different categories. The first model is a classical machine learning model based on *logistic regression*, and the second model is a transformer-based model based on the *DistilBERT architecture*.")
        st.write("The app uses Streamlit for the web interface and various libraries for text processing and normalization, such as NLTK, Contractions, and SpaCy. The logistic regression model and the CountVectorizer used for feature extraction are saved in joblib files, and the DistilBERT model is saved in TensorFlow Lite format.")
        st.write("You can select one of the two models and enter a sentence in the text input. The app will then normalize the sentence, process it according to the selected model, and output the predicted label and the corresponding probability.")
        st.divider()
    

        # Create two columns to make the model selector narrower
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_model = st.selectbox("Select a model:", model_options)
                    

        # Description of the selected model
        if selected_model == "Classical Machine Learning for NLP":
            st.write("This is a simple model that helps in classifying data into different categories, in this case, it is designed to handle multiple categories. It uses a mathematical approach called logistic regression and adjusts its predictions based on past data. This model is designed to handle text data and processes it to make it ready for analysis. The text is transformed into numerical values, and the model uses these values to make predictions. This model is not too complex and is easy on computing resources. For more information, please refer to the scikit-learn documentation for [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).")
        elif selected_model == "Transformer Machine Learning for NLP":
            st.write("DistilBERT is a smaller, faster version of the popular BERT model. It has been trained by the company Hugging Face to perform text classification tasks, such as sentiment analysis and named entity recognition. The model uses an advanced technique called 'distillation' to reduce its size and computational requirements, while still maintaining its accuracy. This makes it a good choice for businesses and organizations that need to classify text data but have limited resources. For more information, please refer to the Hugging-Face documentation for [DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert#distilbert).")
        else:
            st.write("Please select one model.")

        # Input section
        input_text = st.text_area("Enter a sentence (Max 85 words):",
                                  height=100
                                  )

        # Output section
        if input_text:
            words = input_text.split()
            if len(words) > 85:
                st.error("The number of words cannot be greater than 85.")
            else:
                if input_text and selected_model != "Select a model":
                    if selected_model == "Classical Machine Learning for NLP":
                        top3_labels, top3_probs, normalized_text  = predict_label_model1(input_text)
                    elif selected_model == "Transformer Machine Learning for NLP":
                        top3_labels, top3_probs, normalized_text  = predict_label_model2(input_text)
                    else:
                        top3_labels, top3_probs = None, None
                    if top3_labels is not None and top3_probs is not None:
                        plot_probabilities(top3_labels, top3_probs, f"{selected_model} Probabilities")
                        st.write("`Input text:`", input_text)
                        st.write("`Normalized text:`", normalized_text)


with tab5:
    with st.container():
        # About us section
        st.write("Project owners:")

        # Owner 1
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                
                st.image("data/images/Adri.jpeg", width=100)
                
            with col2:
                st.write("[Adriano Persegani Daguzan](https://www.linkedin.com/in/adriano-persegani/)")
                st.write("Data Scientist offering 6+ years in hospitality leadership, backed by a solid educational foundation in computer science. Adept at applying diverse expertise for innovative solutions.")

        # Owner 2
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                
                st.image("data/images/Tuyen.jpeg", width=100)

            with col2:
                st.write("[Tuyen Nguyen Thi](https://www.linkedin.com/in/tuyen-nguyen-thi-7576967b/)")
                st.write("Ph.D. in Applied Mathematics (Hamilton-Jacobi equations in non-periodic settings). I also did twice one-year postdocs in a University. Besides, I'm very excited to learn and approach new methods with high applicability.")

        # Owner 3
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                
                st.image("data/images/Yeeun.jpeg", width=100)

            with col2:
                st.write("[Yeeun Kim](https://www.linkedin.com/in/yeeun-kim-bba19b15b/)")
                st.write("Recent PhD in Educational Psychology. Drawing from my humanistic and scientific background, I like to understand people's behavior and develop predictions.")

        # Owner 4
        with st.container():
            col1, col2 = st.columns([1, 2])

            with col1:
                
                st.image("data/images/Ibra.jpeg", width=100)

            with col2:
                st.write("[Ibrahima Ba](https://www.linkedin.com/in/ibrahima-ba-data-scientist-germany/)")
                st.write("Scientific assistant with 3+ years of experience in advanced mathematical modelling development and Optimize traffic on the networks (assignment, route selection, priorities at intersections).")
        st.divider()
        # Partner images, links, and descriptions at the end of the page

        st.write("In partnership with:")

        with st.container():
            col1, col2 = st.columns(2)

            # Partner 1
            with col1:
                gif_html = get_img_with_href('data/images/constructor_learning.png', 'https://learning.constructor.org/')
                st.markdown(gif_html, unsafe_allow_html=True)


            # Partner 2
            with col2:
                gif_html = get_img_with_href('data/images/rolos.png', 'https://rolos.com/')
                st.markdown(gif_html, unsafe_allow_html=True)

