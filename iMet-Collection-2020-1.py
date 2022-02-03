import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2 as cv2
from PIL import Image

st.set_page_config(
    page_title="iMet Collection 2020 Classification",
    page_icon="🎨",
)
raw_label = pd.read_csv('labels.csv')
array = raw_label.values

print(raw_label)
def main():    
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden; }
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown("<div><h2 style='text-align: center; color: 62CDFF; font-family: Century Gothic'> iMet Collection 2020 Classification </h2></div>", unsafe_allow_html=True)
    model = tf.keras.models.load_model('iMet_Collection')

    raw_image = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

    if(raw_image is not None):
        image_file = Image.open(raw_image)
        image_file = np.array(image_file.convert('RGB'))
        st.image(image_file, use_column_width = True)
 
        predict_image = cv2.resize(image_file, (32,32))
        predict_image = predict_image/255
        predict_image = np.expand_dims(predict_image, axis=0)

        # predict = model.predict(predict_image)
        predictions = model.predict(predict_image,verbose = 1)
        print(predictions)

        text = '<p style="font-family:Century Gothic; color:White; font-size: 30px;">Predict Answer is :</p>'
        st.markdown(text, unsafe_allow_html = True)

        count = 0
        check = False

        for i in range(len(predictions[0])):
            if predictions[0][i] > 0.2:
                check = True
                Ans = array[count]
                st.markdown(Ans, unsafe_allow_html = True)
            count += 1
            
        if(check == False):
            nullans = '<p style="text-align: center; color: 62CDFF; font-family: Century Gothic"> Null </p>'
            st.markdown(nullans, unsafe_allow_html = True)

if __name__ == '__main__':    
    main()
