# Live-Emotion-Classifier

This project uses neural networks to train a model to classify 7 human emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise. This model uses transfer learning with VGG16, and two labeled datasets of emotions to classify a person's emotion given one or more facial expressions. This model was utilized with OpenCV in order to produce a live video feed and emotion classification when a person's face is in shot. After recording, a time series displaying a rolling average of the happiness index of the face or crowd is created in order to evaluate audience response during video capture.

## Getting Started
The below has currently only been tested on macbooks running Sierra to Mojave.

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

## Demonstration

(Add demonstration gif and happiness index tracker here)

## Built With

* [OpenCV](https://docs.opencv.org/4.1.0/) - Face detection and live video capture
* [keras](https://keras.io/) - Construction of models
* [sklearn](https://scikit-learn.org/stable/whats_new.html) - Model preparation and evaluation
* [Google CoLabs](https://colab.research.google.com/notebooks/welcome.ipynb) - Environment used for training the models

## Authors

* **Justin Mitchell** - *Initial work* - [GitHub](github.com/jdmitchell0216)
* **Kevin Zhao** - *Initial work* - [GitHub](github.com/kevzha)

