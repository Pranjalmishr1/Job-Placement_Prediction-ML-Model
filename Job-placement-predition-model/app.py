import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import pickle

lg = pickle.load(open('placement.pkl','rb'))

# web app
img = Image.open('Job-Placement-Agency.jpg')
st.image(img,width=650)

st.title("Job Placement Prediciton Model")
input_text= st.text_input(" enter all fertures")
input_list = input_text.split(',')
# First, make sure input_list exists
if input_list:
    # Check if exactly 14 features are provided
    if len(input_list) != 14:
        st.write(f"Error: Please provide all 14 features. You provided {len(input_list)}.")
    else:
        # Convert yes/no to 1/0 and handle empty strings as 0
        input_numeric = []
        for x in input_list:
            if isinstance(x, str):
                x_lower = x.strip().lower()
                if x_lower == 'yes':
                    input_numeric.append(1)
                elif x_lower == 'no' or x_lower == '':
                    input_numeric.append(0)
                else:
                    try:
                        input_numeric.append(float(x_lower))
                    except ValueError:
                        st.write(f"Invalid input: {x}")
                        break
            else:
                input_numeric.append(float(x))

        # Only predict if we successfully converted all 14 features
        if len(input_numeric) == 14:
            np_df = np.asarray(input_numeric, dtype=float)
            prediction = lg.predict(np_df.reshape(1, -1))

            if prediction[0] == 1:
                st.write("This Person Is Placed")
            else:
                st.write("This Person is not Placed")
 # //if input_list:
 #    np_df = np.asarray(input_list, dtype=float)
 #    prediction = lg.predict(np_df.reshape(1, -1))
 #
 #    if prediction[0] == 1:
 #        st.write("This Person Is Placed")
 #    else:
 #        st.write("This Person is not Placed")