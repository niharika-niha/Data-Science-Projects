import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#background-image: url("https://i.redd.it/nflafw75q6a91.jpg"); ("https://cdn.wallpapersafari.com/88/75/cLUQqJ.jpg");

#Styling section
st.markdown("""
    <style>
    
        # [data-testid="stAppViewContainer"]{
  # background-image: url("https://wallpaperbat.com/img/92274-black-and-blue-tech-wallpaper.jpg");
  # background-size: cover;px;
  # box-shadow: inset 0 0 0 2000px rgba(0, 0, 0, 0.6);
        # }
        # .prediction-box {background-color: #ecf0f1; padding: 15px; border-radius: 10px;}
        .number_input {
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
        }
      
    </style>
""", unsafe_allow_html=True)

# Function to make prediction based on user input
def predict(features):
    print("input from User(inside func): ",features)
    # load your trained model 
    model = joblib.load('logistic_model.pkl')
    scaler= joblib.load('scaler.pkl')

    # Use the features as input for prediction
    # Features need to be scaled or transformed similarly to how the model was trained
    features[1:6] = list(scaler.transform([features[1:6]]).flatten())
    
    print("Scaled features(inside func): ",features)
    prediction = model.predict([features])
    print("prediction(inside func): ",prediction)
    pred_prob=model.predict_proba([features])
    print("prediction Probability(inside func): ",pred_prob)
    print("-"*100)
    return [prediction, features, pred_prob]


st.markdown('<h2 style="color: Orange;font-family: Poppins, sans-serif;">The CrashCatcher</h2>', unsafe_allow_html=True)
st.write("Machine Failure Prediction")
#Input features in columns for a cleaner look
col1, col2 = st.columns(2)
with col1:
    type_ = st.selectbox(":blue[Type]", ['High', 'Low', 'Medium'])
    temp_air = st.number_input(":blue[Air temperature [K]]", max_value=350.0, step=0.1, value=None, format="%.1f")
    temp_process = st.number_input(":blue[Process temperature [K]]",  max_value=400.0, step=0.1,value=None, format="%.1f")
    
with col2:
    rotational_speed = st.number_input(":blue[Rotational speed [rpm]]",  max_value=5000.0, step=0.1,value=None, format="%.1f")
    torque = st.number_input(":blue[Torque [Nm]]", max_value=500.0, step=0.1,value=None, format="%.1f")
    tool_wear = st.number_input(":blue[Tool wear [min]]",  max_value=500.0, step=0.1,value=None, format="%.1f")
selected = st.pills(":blue[Which of the components below are experiencing failure?]", ["TWF", "HDF", "PWF","OSF","RNF"], selection_mode="multi")
if selected:
    st.markdown(f"You selected the following components as failed:  **{', '.join(selected)}**")
    
TWF = int('1') if "TWF" in selected else int('0')
HDF = int('1') if "HDF" in selected else int('0')
PWF = int('1') if "PWF" in selected else int('0')
OSF = int('1') if "OSF" in selected else int('0')
RNF = int('1') if "RNF" in selected else int('0')
type_input=0
if type_=="High":type_input=0;
elif type_=="Low":ype_input=1;
elif type_=="Medium":ype_input=2;

#features names
cols=['Type','Air temperature [K]', 'Process temperature [K]','Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]','TWF', 'HDF', 'PWF', 'OSF', 'RNF']

# Once the user fills in the inputs, create a list of the feature values
# user_input = [0, 300.5, 309.9, 1397, 45.9, 210, 1, 0, 0, 0, 0] #should give 1 as prediction value- fail
# user_input =[1,300.8,310.3,1538.0,36.1,198,0,0,0,0,0] # should give 0 as output - success
user_input = [type_input, temp_air, temp_process, rotational_speed, torque, tool_wear, TWF, HDF, PWF, OSF, RNF]
print("outside-\n User input: ",user_input)

user_input_data=pd.DataFrame(np.array(user_input).reshape(1, -1), columns=cols)
# Add a button to trigger the prediction
# Check if the user has entered a number



if st.button("🔮 Predict", key="predict"):
    if (len([i for i in user_input if i is not None]))!=11:
        print("1st")
        st.warning("📝 **Note:** Please make sure all fields are filled correctly before submitting.")
    else:       
        
        prediction,features,pred_prob = predict(user_input) #calling predict function
        print("outside-\n prediction: ",prediction,"\nfeatures: ",features,"\npred_prob: ",pred_prob)

        features_data = pd.DataFrame(np.array(features).reshape(1, -1), columns=cols)
        pred_prob_data=pd.DataFrame(np.array(pred_prob).reshape(1, -1), columns=["success","failure"])
        #The first value of the pred_prob array corresponds to Class 0|"negative"|"failure" class.
        
        # st.markdown('<div class="prediction-box">Prediction: {}</div>'.format("Uh-oh, Prediction Value is 1.The machine is currently operating at a high chance of failure. It is recommended to monitor its performance closely and take necessary precautions...🛠️"
                             # if prediction[0] == 1 else "Good news! Prediction Value is 0.The machine is as reliable as your morning coffee—no signs of trouble ahead! 😊"), unsafe_allow_html=True)


        # results in tabs
        tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
        
        tab1.subheader("plotting scaled values of parameter using scatterplot 📊  ")
        

        # Create scatter plot
        df=features_data.T    
        fig = px.scatter(df, x=cols, y=features, 
                         color=features,  # Use 'Value' column for color gradient
                         color_continuous_scale="Turbo",#"Viridis",  # Gradient color scale
                         text=features)
        # Customize layout
        fig.update_traces(textposition='top center')
        fig.update_layout(yaxis_title="Value", xaxis_title="Machine Parameters",coloraxis_showscale=False)

        # Display in Streamlit tab    
        tab1.plotly_chart(fig)



        # Display failure message
        if prediction[0] == 0:st.success("✅ Good news! The machine is as reliable as your morning coffee—no signs of trouble ahead! 😊")
        else: st.error("⚠ Uh-oh, Machine Failure Detected! Check system parameters and take necessary precautions.🛠️")
        
        # Data in tab2
        tab2.subheader("User Input Data")
        tab2.write(user_input_data)
        tab2.subheader("Scaled Data")
        tab2.write(features_data)    
        tab2.subheader("prediction probabilities")
        tab2.write(pred_prob_data)
        