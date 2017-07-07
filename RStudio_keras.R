if (!require(keras)) {
  devtools::install_github("rstudio/keras")
  library(keras)
}

install_tensorflow()
# To install a version of TensorFlow that takes advantage of Nvidia GPUs if 
# you have the correct CUDA libraries installed.
# install_tensorflow(gpu = TRUE)


# Test keras --------------------------------------------------------------

# loading keras library
library(keras)

# loading the keras inbuilt mnist dataset
data <- dataset_mnist()

# separating train and test file
train_x <- data$train$x
train_y <- data$train$y
test_x  <- data$test$x
test_y  <- data$test$y
rm(data)

# converting a 2D array into a 1D array for feeding into the MLP and 
# normalising the matrix
train_x <- array(train_x, dim = c(dim(train_x)[1], 
                                  prod(dim(train_x)[-1]))) / 255
test_x  <- array(test_x, dim = c(dim(test_x)[1], 
                                 prod(dim(test_x)[-1]))) / 255

# converting the target variable to once hot encoded vectors using keras 
# inbuilt function
train_y <- to_categorical(train_y, 10)
test_y  <- to_categorical(test_y,  10)

# defining a keras sequential model
model <- keras_model_sequential()

# defining the model with:
#   1 input layer[784 neurons], 
#   1 hidden layer[784 neurons] with dropout rate 0.4 and 
#   1 output layer[10 neurons]
# i.e number of digits from 0 to 9
model %>%
  layer_dense(units = 784, input_shape = 784) %>%
  layer_dropout(rate = 0.4)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 10) %>%
  layer_activation(activation = 'softmax')

# compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# fitting the model on the training dataset
model %>% fit(train_x, train_y, epochs = 100, batch_size = 128)

# Evaluating model on the cross validation dataset
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)


# RStudio Example ---------------------------------------------------------

# https://rstudio.github.io/keras/


# The core data structure of Keras is a model, a way to organize layers. The simplest type of model is the Sequential model, a linear stack of layers. For more complex architectures, you should use the Keras functional API, which allows to build arbitrary graphs of layers.
# 
# Define a sequential model:
library(keras)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.2) %>% 
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 10, activation = 'softmax')

# Print a summary of the model’s structure using the summary() function:
summary(model)

# Compile the model with appropriate loss function, optimizer, and metrics:
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(lr = 0.02),
  metrics = c('accuracy')
)

# You can now iterate on your training data in batches 
# (x_train and y_train are R matrices):
history <- model %>% fit(
  # x_train, y_train, 
  train_x, train_y, # From the example above
  epochs = 20, batch_size = 128, 
  validation_split = 0.2
)

# Plot loss and accuracy metrics from training:
plot(history)

# Evaluate your performance on test data:
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)

# Generate predictions on new data:
classes <- model %>% predict(test_x, batch_size = 128)
