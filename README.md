# Live-Emotion-Classifier

This project trains several convolutional neural networks to classify 7 human emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise. This model uses transfer learning with VGG16, and two labeled datasets containing facial expressions to classify a person's emotion given one or more facial expressions. This model was used with OpenCV's face detection using Haar Cascades in order to produce a live video feed and emotion classification when 1 to 6 people's faces are in shot. After recording, a time series displaying a rolling average of the happiness index of the face or crowd is created in order to evaluate audience response during video capture.

## Getting Started
To download and use the final product, follow the instructions below. Currently, everything runs successfully on macbooks running Sierra to Mojave using the built in camera. The program has been capped to read 6 faces at once, which is near the number of faces at which point an early 2013 macbook pro with 16GB memory begins to struggle to keep up. For situations that require additional faces to be evaluated, it is advised to either perform the classifications after recording or to use a computer with more processing power.

### Prerequisites
All of the code is run in python 3.7.1.
The following libraries are needed: keras, opencv-python, pandas, numpy, matplotlib.

They can be installed by typing the code below into the terminal.

```
pip install <library name>
```

### Installing

To get the emotion classifier running, follow the steps below.

Download or clone the repository. If you choose to clone, cd into your desired directory and run the line below in terminal.

```
git clone https://github.com/jdmitchell0216/Live-Emotion-Classifier.git
```

While in the directory containing all of the files you just downloaded, enter the line below in terminal to start the live emotion classifier.

```
python3 video_capture_emotion_final.py
```


## Running the tests
If everything works correctly, running the .py file should bring up a video feed that displays the emotion detected for a face above the face it is predicting for. To quit the video capture, press q twice. After quitting, a happiness index numpy array and timestamped plot will be saved into the working directory. If the audience is recorded separately, the recording can be matched up with the time series plot to gauge audience sentiment.

On first runs and in rare cases thereafter, you may run into an error related to cv2.cvtColor. It is likely due to an issue in connecting to your computer's camera. If you have already granted permission to access the camera (which is usually done through a popup window after the first attempt to run the file), try restarting your computer. If the problem still persists, it is likely due to line 35 in the .py file and the cv2.VideoCapture argument may have to be adjusted.

## Demonstration

(Add demonstration gif and happiness index tracker here)

## Built With

* [OpenCV 4.1.0](https://docs.opencv.org/4.1.0/) - Face detection and live video capture
* [keras 2.2.4](https://keras.io/) - Construction of models
* [sklearn 0.20.1](https://scikit-learn.org/stable/whats_new.html) - Model preparation and evaluation
* [Google CoLabs](https://colab.research.google.com/notebooks/welcome.ipynb) - Environment used for training the models

### Version Specifications
* numpy 1.14.5
* pandas 0.24.2
* matplotlib 3.0.3

## Authors

* **Justin Mitchell** - *Initial work* - [GitHub](github.com/jdmitchell0216)
* **Kevin Zhao** - *Initial work* - [GitHub](github.com/kevzha)
