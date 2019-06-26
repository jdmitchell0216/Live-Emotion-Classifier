# Live-Emotion-Classifier
Justin Mitchel, Kevin Zhao

This project uses neural networks to train a model to classify 7 human emotions: anger, disgust, fear, happiness, neutral, sadness, and surprise. This model uses transfer learning with VGG16, and two labeled datasets of emotions to classify a person's emotion given one or more facial expressions. This model was utilized with OpenCV in order to produce a live video feed and emotion classification when a person's face is in shot. After recording, a time series displaying a rolling average of the happiness index of the face or crowd is created in order to evaluate audience response during video capture.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
All of the code is run in python 3.7.1.
The following libraries are needed: keras, cv2, pandas, numpy, matplotlib.

They can be installed by typing the code below into the terminal.

```
pip install <library name>
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [OpenCV](https://docs.opencv.org/4.1.0/) - Face detection and live video capture
* [keras](https://keras.io/) - Construction of models
* [sklearn](https://scikit-learn.org/stable/whats_new.html) - Model preparation and evaluation
* [Google CoLabs](https://colab.research.google.com/notebooks/welcome.ipynb) - Environment used for training the models

## Authors

* **Justin Mitchell** - *Initial work* - [GitHub](github.com/jdmitchell0216)
* **Kevin Zhao** - *Initial work* - [GitHub](github.com/kevzha)

