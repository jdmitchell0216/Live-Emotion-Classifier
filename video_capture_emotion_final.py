# Import
import numpy as np
import cv2
from keras.models import model_from_json
import time
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Initialize a record of in-camera happiness
happy_tracker = []
# Initialize dictionary for assigning emotions to model predictions
emotion_dict={0:'anger',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
# Initialize dictionary for obtaining a color index value for the combined model's prediction
emotion_to_index={'anger':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5,'surprise':6}
# Initialize dictionary for using different font colors for each emotion
color_dict = {0:(59,59,238),1:(0,205,102),2:(30,30,30),3:(0,215,255),4:(255,255,255),5:(238,134,28),6:(255,102,224)}

# Load first model
with open('model9.json', 'r') as f:
    loaded_model = model_from_json(f.read())

# Load first model's weights
loaded_model.load_weights('model_weights9.h5')

# Load second model
with open('model10.json', 'r') as f:
    loaded_model3 = model_from_json(f.read())

# Load second model's weights
loaded_model3.load_weights('model_weights10.h5')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load in opencv's Haar Cascade face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start timer so that "happy tracker" can use timestamps
begin = time.time()

# Begin detection
while(True):
    # Update frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find face positions in current frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Start timer and check for limiting rate of calculations. May not be necessary if number of faces is limited
    start = time.time()
    check = True

    # Limit number of faces to use for each frame. My macbook could handle about 8 at a time before problems occurred
    if len(faces) > 5:
        faces = faces[:5]

    # Prepare images of detected faces for model, classify the emotion of each, and update happy_tracker
    for (x,y,w,h) in faces:
        # Slightly decrease the size of the window determined by the Haar Cascade to match the typical dimensions used in the training of the models
        crop_img = gray[int(y+0.05*h):int(y+0.95*h), int(x+0.05*w):int(x+0.95*w)]

        # Get dimensions of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resize face images to (1,48,48,3) to fit model inputs
        small = cv2.resize(crop_img, dsize = (48,48))
        image3D = np.expand_dims(small,axis = 0)
        image4D = np.expand_dims(image3D, axis = 3)
        image4D3 = np.repeat(image4D, 3, axis=3)

        # Make predictions on each face
        if time.time()-start >= 0.02 or check == True: #make delay higher or face limiter lower if the computer is struggling to keep up
            # Classify emotion with first model
            emotions_prob = loaded_model.predict(image4D3)[0]
            # Assign a 1 to the most likely emotion
            listt = [1 if metric == emotions_prob.max() else 0 for metric in emotions_prob]
            # Get index of most likely emotion
            emotion_index = listt.index(1)
            # Convert emotion index to emotion name
            emotion = emotion_dict[emotion_index]
            start = time.time()
            check == False

            # Classify emotion with second model using the same process as the first model
            emotions_prob3 = loaded_model3.predict(image4D3)[0]
            listt3 = [1 if metric == emotions_prob3.max() else 0 for metric in emotions_prob3]
            emotion_index3 = listt3.index(1)
            emotion3 = emotion_dict[emotion_index3]
            
            # Rule based algorithm for combining both models' predictions into one (optimized empirically)
            if emotion == emotion3:
                emotion_fin = emotion
            elif emotion == 'happy' or emotion3 == 'happy':
                emotion_fin = 'happy'
            elif emotion == 'anger' or emotion3 == 'anger':
                emotion_fin = 'anger'
            elif emotion == 'fear':
                emotion_fin = 'fear'
            elif emotion3 == 'neutral':
                emotion_fin = 'neutral'
            elif emotion == 'sad':
                emotion_fin = 'sad'
            elif emotion3 == 'disgust' or emotion == 'disgust':
                emotion_fin = 'disgust'
            elif emotion3 == 'sad':
                emotion_fin = 'sad'
            else:
                emotion_fin = 'neutral'

        # Add a one if the final prediction is "happy". Add a zero if it isn't "happy". Include a timestamp for each prediction
        happy_tracker.append(np.array([time.time()-begin, emotion_fin == 'happy']))

        # Label each face with its emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_placement  = (int(x + w/3),int(y - h/4))
        fontScale = 1
        fontColor = color_dict[emotion_to_index[emotion_fin]]
        lineType = 4
    
        cv2.putText(frame, 
            '{}'.format(f'{emotion_fin}'), 
            text_placement, 
            font, 
            fontScale,
            fontColor,
            lineType)

        # Display frame with labels
    cv2.imshow('frame',frame)

    # End capture if "q" is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Save happy_tracker
np.save('./happy_tracker_test',np.array(happy_tracker))

# Quit video capture
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load in happy_tracker
tracker = np.load('happy_tracker_test.npy')
df = pd.DataFrame(tracker)
# Convert index to datetime
df[0]=df[0].apply(lambda x: time.ctime(int(x)))
df.set_index(0, inplace = True)
# Group each entry by second, take the sum of each group, and compute a centered rolling average with a window size of 6.
# Currently, this is computing the gross happiness because the zeros have no effect on the sum. If the percentage happiness is desired,
# divide the sum-by-second by the count-by-second. If the net happiness is desired, subtract (count-by-second - sum-by-second) from
# sum-by-second
df2=df.groupby(by=df.index).sum().rolling(6, center=True).mean()
#Create a graph of the happiness time series and save as "plot_test"
plt.xticks(np.linspace(0,len(df2),10), np.round(np.linspace(0,len(df2),10),1))
plt.xlabel(xlabel= 'Time Elapsed')
plt.ylabel(ylabel= 'Happiness Index')
plt.title('Happiness Time Series')
plt.plot(df2.index, df2[1])
plt.savefig('./plot_test')