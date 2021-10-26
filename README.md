# CNN-for-lamb-wave-modes-recognizing-and-extracting
 Here is the CNN codes for lamb wave vibrating modes extracting. "modeCNN.py" is the network with dropout and "modeCNN1.py" is the network with batch normalization.
 "mode_recognizatino.py" is the codes for front interface.
 With considerable training data, the network would be able to recognize and classify the input images into the classes sorted by user(in "modeCNN.py" or "modeCNN1.py").
 The result of classification will be listed in "modes.txt".(The files won't be overwritted, so empty it each time you finished a classification task if you don't need to list all the result in one file.)

# Environment
- Python 3.7.0
- Keras
- Numpy
- OpenCV
