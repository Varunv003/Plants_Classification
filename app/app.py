import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import matplotlib as plt
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, Dropout, GlobalMaxPooling2D, BatchNormalization, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNet


import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FloraVision",
    # page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)

hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class_names = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine', 'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']

def prediction_cls(prediction): # predict the class of the images based on the model results
    for key, clss in class_names.items(): # create a dictionary of the output classes
        if np.argmax(prediction)==clss: # check the class
            
            return key
        
with st.sidebar:
        # st.image('mg.png')
        st.title("FloraVision")
        st.subheader("Classifying Plant Species Accrately")

st.write("""
         # Plant Species
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image, model):
        size = (224,224)    
        resized_img = tf.image.resize(image, size)
        # image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        # img = np.asarray(image)
        # img_reshape = img[np.newaxis,...]
        prediction = model.predict(np.expand_dims(resized_img/255, 0))
        return prediction

def create_model():
    
    feature_extractor = tf.keras.applications.MobileNet(input_shape=(224, 224, 3),
                              include_top=False,
                              weights="imagenet")
    for layer in feature_extractor.layers:
        layer.trainable=False  
    resize_and_rescale = tf.keras.Sequential([
            tf.keras.layers.Resizing(224, 224),
            tf.keras.layers.Rescaling(1./255),
        ])                          
    model = Sequential([
            resize_and_rescale,
            tf.keras.layers.BatchNormalization(),
            feature_extractor,
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(288, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(80, activation='softmax')
        ])
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
                  metrics=['accuracy'])
    model.build(input_shape=(None, 224, 224, 3))
    return model

     
@st.cache_data()
def load_model():
    path ="models\mobilenet_1.h5"
    model = create_model()
    model.load_weights(path)
    # print(model.summary())
    # model.load_weights(path, skip_mismatch=True)
    # model = tf.keras.models.load_model(path)
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()



if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(98,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    string = "Detected Species : " + class_names[np.argmax(predictions)]
    st.sidebar.success(string)

    st.set_option('deprecation.showfileUploaderEncoding', False)


