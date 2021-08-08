
import streamlit as st
import pandas as pd
import numpy as np
# import pickle

# %matplotlib inline

# import tensorflow_datasets as tfds

import os
# import cv2
# import keras
# import numpy as np
# import pandas as pd
# import seaborn as sn
import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt
# from keras.optimizers import Adam
# from keras.models import Sequential
# from sklearn.metrics import classification_report,confusion_matrix
# from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout

from PIL import Image, ImageOps
#
# import urllib.request
# import urllib.error


st.title("""
This model classifies Singaporean Food.
""")

st.sidebar.header('Categories')
st.sidebar.write(['Dendang Paru', 'Dosa', 'Frog Leg Porridge', 'Goreng Pisang', 'Green Bean Soup', 'Gulai Daun Ubi', 'Hainanese Curry Rice', 'Mee Goreng', 'Meepok', 'Pecel_Lele', 'Rawon', 'Roti_Prata', 'Sambal_Lala', 'Shao Mai', 'Singapore_Sling', 'Tumpeng', 'Vegetarian_Beehoon', 'crab bee hoon'])

# st.sidebar.markdown("""
# ["Carrot Cake","Char Kway Teow",  "Mee Siam", "Roti Prata"]
# """)


#
# url = 'https://github.com/customized/mamamm/blob/16f0ed2ff0c680923584179c948d1dbc874b083c/eff_mod_Food_Localization_2.h5'
# filename = url.split('/')[-1]
# urllib.request.urlretrieve(url, filename)
@st.cache(suppress_st_warning=True)
def load_my_model():
    model2 = tf.keras.models.load_model("model") #for gke
    return model2

size = (224, 224)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])



if uploaded_file is not None:

    up_image = Image.open(uploaded_file)
    st.image(up_image, caption='Uploaded Image.', use_column_width=True)

    upimage = ImageOps.fit(up_image, (224,224))

    img = tf.keras.preprocessing.image.img_to_array(up_image)

    img = tf.keras.preprocessing.image.smart_resize(img, size)
    img = img.reshape(1,224,224,3)


    image2 =img


    st.write("")
    st.write("Classifying...")

    # model2 = tf.keras.models.load_model('/content/gdrive/MyDrive/Colab_notebooks/eff_mod_Food_Localization_2.h5')
    # model2 = tf.keras.models.load_model("") #for streamlit share
    # model2 = tf.keras.models.load_model("model") #for gke
    load_my_model()
    predictions_sg = model2.predict(image2)

    classes_sg = np.argmax(predictions_sg, axis = 1)

    cat_list = ['Dendang Paru', 'Dosa', 'Frog Leg Porridge', 'Goreng Pisang', 'Green Bean Soup', 'Gulai Daun Ubi', 'Hainanese Curry Rice', 'Mee Goreng', 'Meepok', 'Pecel_Lele', 'Rawon', 'Roti_Prata', 'Sambal_Lala', 'Shao Mai', 'Singapore_Sling', 'Tumpeng', 'Vegetarian_Beehoon', 'crab bee hoon']

    st.header(cat_list[int(classes_sg)])
