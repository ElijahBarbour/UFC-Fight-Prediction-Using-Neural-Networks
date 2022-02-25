import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os
from tensorflow import keras


def load_data_1():
    with open('saved_steps.pkl', 'rb') as file:
        data_1 = pickle.load(file)
    return data_1

def load_data_2():
    with open('saved_steps_NN.pkl', 'rb') as file:
        data_2 = pickle.load(file)
    return data_2

def prediction_function(X, bImage, rImage, Blue_Fighter, Red_Fighter):
    Winner = model_nn.predict_on_batch(X)
    #print(Winner)
    #proba = model_nn.predict_proba(X)
    #if proba[0][0] > proba[0][1]: proba = proba[0][0]*100
    #else: proba = proba[0][1]*100
#Howd i get the >=0? because i cross referenced the output of the original 
#Machine Learning Model to predict
#I used Middleweight Garreth Mclellan vs Vik Grujic

    st.title("")
    st.title("")
    if Winner[0] >= 0:
        st.write("""## Winner:""")
        st.image(bImage)
        st.title("")
        st.write("""### {}""".format(Blue_Fighter))
        #st.write("""#### Prediction Probability {} wins: {:.2f}%""".format(Blue_Fighter, proba))
    else:
        st.write("""## Winner:""")
        st.image(rImage)
        st.title("")
        st.write("""### {}""".format(Red_Fighter))
        #st.write("""#### Prediction Probability {} wins: {:.2f}%""".format(Red_Fighter, proba))
                
UFC_fighter_photo_loc = '../../Binary_Classification_UFC_Dataset/UFC_Fighters_Photos/UFCFightersPhotos'#os.getcwd()+"/UFC_Fighters_Photos/UFCFightersPhotos"

image = Image

data_1 = load_data_1()
data_2 = load_data_2()

UFC_NN_model_loc = os.getcwd()+'//UFC_NN_model'
model_nn = keras.models.load_model(UFC_NN_model_loc)

weight_classes = data_1["weight_classes"]
fighter_classes = data_1["fighter_classes"]
fighter_list = data_1["fighter_list"]
fighter_stats = data_1["fighter_stats"]

training_model_acc = data_2["training_model_acc"]
test_model_acc = data_2["test_model_acc"]

def show_page():
    training_m_acc = training_model_acc * 100
    test_m_acc = test_model_acc * 100
    st.title("UFC Fight Prediciton Using Neural Networks")
    #st.write(os.getcwd())
    st.write("""#### Training Model Accuracy: {:.2f}%""".format(training_m_acc))
    st.write("""#### Test Model Accuracy: {:.2f}%""".format(test_m_acc))
    st.title("")
    st.write("""### Choose Your Weight Class""")
    st.write("##")
    weight_class = st.selectbox(label = "Weight Classes", options = weight_classes)
    weight_class = weight_class.replace("\'", "")
    weight_class = weight_class.replace(" ", "_")
    weight_class = weight_class.lower()
    
    fighters_ = fighter_classes[weight_class]
    fighters_ = set(fighters_.dropna())
    st.title("")
    st.title("")
    col1, col2, col3 = st.columns(3)
    with col1:
        Red_Fighter = st.selectbox(label = "Select Red Fighter", options = fighters_)
        rFighter = Red_Fighter.replace(" ", "-")
        rFighter_loc = UFC_fighter_photo_loc+'/'+rFighter+".jpg"
        #st.write(rFighter_loc)
        try: rImage = Image.open(rFighter_loc)
        except:
            if(weight_class == 'womens_strawweight' or weight_class == 'womens_flyweight' or weight_class == 'womens_bantamweight' or weight_class == 'womens_featherweight'):
                rFighter_loc = UFC_fighter_photo_loc+'/'+"Default-G.jpg"
                rImage = Image.open(rFighter_loc)
            else:
                rFighter_loc = UFC_fighter_photo_loc+'/'+"Default-B.jpg"
                rImage = Image.open(rFighter_loc)

        st.image(rImage)

    with col3:
        Blue_Fighter = st.selectbox(label = "Select Blue Fighter", options = fighters_)
        bFighter = Blue_Fighter.replace(" ", "-")
        bFighter_loc = UFC_fighter_photo_loc+'/'+bFighter+".jpg"
        try:
            bImage = Image.open(bFighter_loc)
        except:
            if(weight_class == 'womens_strawweight' or weight_class == 'womens_flyweight' or weight_class == 'womens_bantamweight' or weight_class == 'womens_featherweight'):
                bFighter_loc = UFC_fighter_photo_loc+'/'+"Default-G.jpg"
                bImage = Image.open(bFighter_loc)
            else:
                bFighter_loc = UFC_fighter_photo_loc+'/'+"Default-B.jpg"
                bImage = Image.open(bFighter_loc)

        st.image(bImage)

    with col2:
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        st.title("")
        ok = st.button("Predict")
        if ok:
            if Red_Fighter == Blue_Fighter:
                st.write("Red Fighter and Blue Fighter should not be the same person")
            elif Red_Fighter == ' ':
                st.write("Please choose Red Fighter")
            elif Blue_Fighter == ' ':
                st.write("Please choose Blue Fighter")
            elif Red_Fighter == ' ' and Blue_Fighter == ' ':
                st.write("Please choose Red and Blue Fighter")
            else:
                b_index = fighter_stats.index[fighter_stats['Name'] == Blue_Fighter]
                b_index = b_index[0]

                r_index = fighter_stats.index[fighter_stats['Name'] == Red_Fighter]
                r_index = r_index[0]

#Why Blue fighter - Red Fighter? The dataset says that every metric measured in there is BF - RF

                win_streak_diff = fighter_stats['current_win_streak'][b_index] - fighter_stats['current_win_streak'][r_index]
                #loss_diff = fighter_stats['losses'][b_index] - fighter_stats['losses'][r_index]
                #round_count_diff = fighter_stats['total_rounds_fought'][b_index] - fighter_stats['total_rounds_fought'][r_index]
                #reach_diff = fighter_stats['Reach_cms'][b_index] - fighter_stats['Reach_cms'][r_index]
                age_diff = fighter_stats['age'][b_index] - fighter_stats['age'][r_index]
                avg_TD_landed_diff = fighter_stats['avg_TD_landed'][b_index] - fighter_stats['avg_TD_landed'][r_index]
                #sig_str_diff = fighter_stats['sig_strikes_landed'][b_index] - fighter_stats['sig_strikes_landed'][r_index]



                X = np.array([[age_diff, win_streak_diff, avg_TD_landed_diff]])

                try:
                    prediction_function(X, bImage, rImage, Blue_Fighter, Red_Fighter)
                except:
                    st.write("""## I am sorry!""")
                    st.write("""### We have encountered an error when processing these fighters.""")
                    st.write("""### Please Try a Different Red and/or Blue Fighter""")

show_page()