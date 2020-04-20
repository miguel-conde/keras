library(keras)
library(tidyverse)


# Data --------------------------------------------------------------------


# loading the keras inbuilt mnist dataset

mnist <- dataset_mnist()

c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist


# Model 1 -----------------------------------------------------------------

## Instantiating a small convnet
model <- keras_model_sequential() %>%
  # shape (image_height, image_width, image_channels) 
  # (not including the batch dimension).
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

# The output of every layer_conv_2d and layer_max_pooling_2d is a 3D tensor of 
# shape (height, width, channels). The width and height dimensions tend to 
# shrink as you go deeper in the network. The number of channels is controlled 
# by the first argument passed to the layer_conv_2d (32 or 64).
model

## Adding a classifier on top of the convnet

# The next step is to feed the last output tensor (of shape (3, 3, 64)) into a 
# densely connected classifier network: 
# a stack of dense layers. These classifiers process vectors, which are 1D, 
# whereas the current output is a 3D tensor. First we have to flatten the 3D 
# outputs to 1D, and then add a few dense layers on top.
model <- model %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

model

## Training the convnet on MNIST images

train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255

train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size=64
)

results <- model %>% evaluate(test_images, test_labels)

results


# Training a convnet from scratch on a small dataset ----------------------

## Copying images to train, validation, and test directories

original_dataset_dir <- here::here("data", "Downloads", "kaggle_original_data")
base_dir <- here::here("data", "Downloads", "cats_and_dogs_small")

dir.create(base_dir)

train_dir <- file.path(base_dir, "train")
dir.create(train_dir)

validation_dir <- file.path(base_dir, "validation")
dir.create(validation_dir)

test_dir <- file.path(base_dir, "test")
dir.create(test_dir)

train_cats_dir <- file.path(train_dir, "cats")
dir.create(train_cats_dir)

train_dogs_dir <- file.path(train_dir, "dogs")
dir.create(train_dogs_dir)

validation_cats_dir <- file.path(validation_dir, "cats")
dir.create(validation_cats_dir)

validation_dogs_dir <- file.path(validation_dir, "dogs")
dir.create(validation_dogs_dir)

test_cats_dir <- file.path(test_dir, "cats")
dir.create(test_cats_dir)

test_dogs_dir <- file.path(test_dir, "dogs")
dir.create(test_dogs_dir)

fnames <- paste0("cat.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(train_cats_dir))

fnames <- paste0("cat.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(validation_cats_dir))

fnames <- paste0("cat.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(test_cats_dir))

fnames <- paste0("dog.", 1:1000, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(train_dogs_dir))

fnames <- paste0("dog.", 1001:1500, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(validation_dogs_dir))

fnames <- paste0("dog.", 1501:2000, ".jpg")
file.copy(file.path(original_dataset_dir, "train", fnames),
          file.path(test_dogs_dir))

cat("total training cat images:", length(list.files(train_cats_dir)), "\n")

cat("total training dog images:", length(list.files(train_dogs_dir)), "\n")

cat("total validation cat images:", length(list.files(validation_cats_dir)), "\n")

cat("total validation dog images:", length(list.files(validation_dogs_dir)), "\n")

cat("total test cat images:", length(list.files(test_cats_dir)), "\n")

cat("total test dog images:", length(list.files(test_dogs_dir)), "\n")

## Instantiating a small convnet for cats vs. dogs classification

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

## Configuring the model for training

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

## Using image_data_generator to read images from directories
#
# The image_data_generator() function, which can automatically turn image files 
# on disk into batches of pre-processed tensors
#   1 - Read the picture files.
#   2 - Decode the JPEG content to RGB grids of pixels.
#   3 - Convert these into floating-point tensors.
#   4 - Rescale the pixel values (between 0 and 255) to the [0, 1] interval 
#      (as you know, neural networks prefer to deal with small input values).
train_datagen <- image_data_generator(rescale = 1/255)
validation_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  validation_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

## Displaying a batch of data and labels

# Note that the generator yields these batches indefinitely: it loops endlessly 
# over the images in the target folder.

batch <- generator_next(train_generator)
str(batch)

## Fitting the model using a batch generator

# The  fit_generator function (the equivalent of fit for data generators)
#   - It expects as its first argument a generator that will yield batches of 
#     inputs and targets indefinitely. 
#        - Because the data is being generated endlessly, the generator needs 
#          to know how many samples to draw from the generator before declaring
#          an epoch over. 
#             - This is the role of the steps_per_epoch argument: after having 
#               drawn steps_per_epoch batches from the generator — that is, 
#               after having run for steps_per_epoch gradient descent steps — 
#               the fitting process will go to the next epoch.
#             - In this case, batches are 20-samples large, so it will take 100 
#               batches until you see your target of 2,000 samples.
#    - When using fit_generator, you can pass a validation_data argument, much 
#      as with the fit function. 
#        - It’s important to note that this argument is allowed to be a data
#          generator, but it could also be a list of arrays. 
#             - If you pass a generator as validation_data, then this generator 
#               is expected to yield batches of validation data endlessly; thus 
#               you should also specify the validation_steps argument, which 
#               tells the process how many batches to draw from the validation 
#               generator for evaluation.

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

## Saving the model

model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

## Displaying curves of loss and accuracy during training

plot(history)


# Using data augmentation -------------------------------------------------

# Overfitting is caused by having too few samples to learn from, rendering you 
# unable to train a model that can generalize to new data. 
#
# Given infinite data, your model would be exposed to every possible aspect of 
# the data distribution at hand: you would never overfit. 
#
# Data augmentation takes the approach of generating more training data from
# existing training samples, by augmenting the samples via a number of random
# transformations that yield believable-looking images. 
#
#   The goal is that at training time, your model will never see the exact 
#   same picture twice. 
# 
# This helps expose the model to more aspects of the data and generalize better.

## Setting up a data augmentation configuration via image_data_generator

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

## Displaying some randomly augmented training images

fnames <- list.files(train_cats_dir, full.names = TRUE)

img_path <- fnames[[3]]

img <- image_load(img_path, target_size = c(150, 150))
img_array <- image_to_array(img)
img_array <- array_reshape(img_array, c(1, 150, 150, 3))

augmentation_generator <- flow_images_from_data(
  img_array,
  generator = datagen,
  batch_size = 1
)

orig_op <- par()
op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0))

for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
}

par(op)

## Defining a new convnet that includes dropout
#
# If you train a new network using this data-augmentation configuration, the 
# network will never see the same input twice. But the inputs it sees are still 
# heavily intercorrelated, because they come from a small number of original 
# images—you can’t produce new information, you can only remix existing 
# information. As such, this may not be enough to completely get rid of 
# overfitting. To further fight overfitting, you’ll also add a dropout
# layer to your model, right before the densely connected classifier.

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("acc")
)

## Training the convnet using data-augmentation generators

datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 32,
  class_mode = "binary"
)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)

## Saving the model

model %>% save_model_hdf5("cats_and_dogs_small_2.h5")

## Displaying curves of loss and accuracy during training

plot(history)


# Using a pretrained convnet ----------------------------------------------

# The VGG16 model, among others, comes prepackaged with Keras. Here’s the list 
# of image-classification models (all pretrained on the ImageNet dataset) that 
# are available as part of Keras:
#
#  - Xception
#  - InceptionV3
#  - ResNet50
#  - VGG16
#  - VGG19
#  - MobileNet

## Instantiating the VGG16 convolutional base

conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(150, 150, 3)
)

conv_base


# The final feature map has shape (4, 4, 512). That’s the feature on top of 
# which you’ll stick a densely connected classifier.
#
# At this point, there are two ways you could proceed:
# 
# - Running the convolutional base over your dataset, recording its output to an 
# array on disk, and then using this data as input to a standalone, densely 
# connected classifier similar to those you saw in part 1 of this book. This 
# solution is fast and cheap to run, because it only requires running the 
# convolutional base once for every input image, and the convolutional base is 
# by far the most expensive part of the pipeline. But for the same reason, this 
# technique won’t allow you to use data augmentation.
# 
# - Extending the model you have (conv_base) by adding dense layers on top, and 
# running the whole thing end to end on the input data. This will allow you to 
# use data augmentation, because every input image goes through the convolutional 
# base every time it’s seen by the model. But for the same reason, this technique 
# is far more expensive than the first.

### FAST FEATURE EXTRACTION WITHOUT DATA AUGMENTATION

## Extracting features using the pretrained convolutional base

base_dir <- here::here("data", "Downloads", "cats_and_dogs_small")

train_dir <- file.path(base_dir, "train")
validation_dir <- file.path(base_dir, "validation")
test_dir <- file.path(base_dir, "test")

datagen <- image_data_generator(rescale = 1/255)

batch_size <- 20

extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150),
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    
    batch <- generator_next(generator)
    
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    
    features_batch <- conv_base %>% predict(inputs_batch)
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      break
  }
  
  list(
    features = features,
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}

train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

## Defining and training the densely connected classifier 

model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu",
              input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

## Plotting the results

plot(history)

## Saving the model

model %>% save_model_hdf5("cats_and_dogs_small_3.h5")
