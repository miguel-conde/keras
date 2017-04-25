# keras
Intro to keras

## Installation (Windows)

[Installation - Keras Documentation](https://keras.io/#installation)

We are asuming we've already installed tensorflow.

0. **On a system terminal.**

If your tensorflow installation is on an anaconda environment, first of all activate 
it.

```
activate tensorflow
```

1. **Install keras on your system**

Go to some temporary directory and execute:

```
git clone git://github.com/fchollet/keras.git
```

This will create a *keras* directory. 

```
cd keras
```

And execute:

```
python setup.py install
```

Now install *scipy* package. You can use:

```
pip install scipy
```

Or use the Anaconda Navigator.

To validate the installation, try executing:

```
python test_keras.py
```

where test_keras.py contains:
```
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

data_dim = 20
nb_classes = 4

model = Sequential()

# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, input_dim=data_dim, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',  
              metrics=["accuracy"])

# generate dummy training data
x_train = np.random.random((1000, data_dim))
y_train = np.random.random((1000, nb_classes))

# generate dummy test data
x_test = np.random.random((100, data_dim))
y_test = np.random.random((100, nb_classes))

model.fit(x_train, y_train,
          nb_epoch=20,
          batch_size=16)

score = model.evaluate(x_test, y_test, batch_size=16)
```


2. **Install *kerasR* R package.**

Open R and type:

```
install.packages("kerasR")
```