library(keras)
library(tidyverse)


# 1 - BINARY CLASSIFICATION -----------------------------------------------


# Data --------------------------------------------------------------------

## Loading the IMDB dataset

imdb <- dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

str(train_data[[1]])
train_labels[[1]]
max(sapply(train_data, max))

word_index <- dataset_imdb_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

## Encoding the integer sequences into a binary matrix

# Input must be a n_samles x n_features matrix.
vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

#  Here, n_features = 10000
x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

str(x_train[1,])

## Converting labels to numeric

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

str(y_train)


# Model 1 -----------------------------------------------------------------

## The model definition

model <- keras_model_sequential() %>%
  # Input shape indicates the shape of the input WITHOUT the first axis or 
  # dimension, i.e., it doesn't include the number of samples
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

## Compiling the model

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

## Configuring the optimizer
## Using custom losses and metrics
# Equivalent
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001),
  loss = loss_binary_crossentropy,
  metrics = metric_binary_accuracy
)

## Setting aside a validation set

val_indices <- 1:10000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

## Training your model

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

## The history object

str(history)
as_tibble(history)

## Plotting the training and validation metrics

plot(history)

# Final Model -------------------------------------------------------------

## Retraining a model from scratch

model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)

results <- model %>% evaluate(x_test, y_test)

results

## Using a trained network to generate predictions on new data

model %>% predict(x_test[1:10,])
model %>% predict_classes(x_test[1:10,])

# MULTICLASS CLASSIFICATION -----------------------------------------------

rm(list = ls()); gc()


# Data --------------------------------------------------------------------

## Loading the Reuters dataset

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters

length(train_data)
length(test_data)

train_data[[1]]

## Decoding a newswires back to text

word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

decoded_newswire <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

train_labels[[1]]

## Encoding the data

# Features

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

# Labels

to_one_hot <- function(labels, dimension = 46) {
  results <- matrix(0, nrow = length(labels), ncol = dimension)
  for (i in 1:length(labels))
    results[i, labels[[i]]] <- 1
  results
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

# Built-in way to do this in Keras
one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)

# Model 1 -----------------------------------------------------------------

## Model definition

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

## Compiling the model

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

## Setting aside a validation set

val_indices <- 1:1000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[-val_indices,]

## Training the model

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

## Plotting the training and validation metrics

plot(history)

# Final Model -------------------------------------------------------------

## Retraining a model from scratch

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(x_train,
                         one_hot_train_labels,
                         epochs = 9,
                         batch_size = 512)

results <- model %>% evaluate(x_test, one_hot_test_labels)

results

test_labels_copy <- test_labels
test_labels_copy <- sample(test_labels_copy)
length(which(test_labels == test_labels_copy)) / length(test_labels)

## Generating predictions for new data

predictions <- model %>% predict(x_test)
sum(predictions[1,])
which.max(predictions[1,])

# If using integer labels (alternative to one hot encodede labels)
# Chamge loss to sparse_categorical_crossentropy
model %>% compile(
  optimizer = "rmsprop",
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

# REGRESSION --------------------------------------------------------------

rm(list = ls()); gc()

# Data --------------------------------------------------------------------

## Loading the Boston housing dataset

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset

str(train_data)
str(test_data)

head(train_data)

str(train_targets)

## Normalizing the data

mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)
test_data <- scale(test_data, center = mean, scale = std)


# Model 1 -----------------------------------------------------------------

## Model definition

build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

## K-fold validation

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(indices, breaks = k, labels = FALSE)
num_epochs <- 100
all_scores <- c()

for (i in 1:k) {
  
  cat("processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
  
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  
  all_scores <- c(all_scores, results$mae)
}

all_scores

mean(all_scores)


# Model 2 -----------------------------------------------------------------

## Saving the validation logs at each fold

num_epochs <- 500
all_mae_histories <- NULL

for (i in 1:k) {
  
  cat("processing fold #", i, "\n")
  
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  model <- build_model()
  
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  
  mae_history <- history$metrics$val_mae
  
  all_mae_histories <- rbind(all_mae_histories, mae_history)
}

## Building the history of successive mean K-fold validation scores

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

## Plotting validation scores

library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()


# Final model -------------------------------------------------------------

## Training the final model

model <- build_model()

model %>% fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 0)

result <- model %>% evaluate(test_data, test_targets)

result
