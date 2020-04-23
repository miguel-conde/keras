library(keras)
library(tidyverse)



# 1 - VECTORISING TEXT ----------------------------------------------------

# 1.1 - One-hot encoding --------------------------------------------------

# The most common, most basic way to turn a token into a vector.
#
# It consists of associating a unique integer index with every word and then
# turning this integer index i into a binary vector of size N (the size of the 
# vocabulary); the vector is all zeros except for the _i_th entry, which is 1.

## Word-level one-hot encoding (toy example)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

token_index <- list()

for (sample in samples)
  for (word in strsplit(sample, " ")[[1]])
    if (!word %in% names(token_index))
      token_index[[word]] <- length(token_index) + 2

max_length <- 10
results <- array(0, dim = c(length(samples),
                            max_length,
                            max(as.integer(token_index))),
                 dimnames = list(sample = NULL,
                                 word_in_seq = NULL,
                                 word_in_dict = c("", names(token_index))
                                 ))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    index <- token_index[[words[[j]]]]
    results[[i, j, index]] <- 1
  }
}

## Character-level one-hot encoding (toy example)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
token_index <- c(1:(length(ascii_tokens)))
names(token_index) <- ascii_tokens

max_length <- 50
results <- array(0, 
                 dim = c(length(samples), 
                            max_length, length(token_index)),
                 dimnames = list(sample = NULL,
                                 char_in_seq = NULL,
                                 char_in_dict = c(names(token_index))
                 ))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  characters <- strsplit(sample, "")[[1]]
  for (j in 1:length(characters)) {
    character <- characters[[j]]
    results[i, j, token_index[[character]]] <- 1
  }
}

## Using Keras for word-level one-hot encoding

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)

sequences <- texts_to_sequences(tokenizer, samples)

one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")

word_index <- tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")

##Word-level one-hot encoding with hashing trick (toy example)

# A variant of one-hot encoding is the so-called one-hot hashing trick, which 
# you can use when the number of unique tokens in your vocabulary is too large 
# to handle explicitly.

library(hashFunction)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# Stores the words as vectors of size 1,000. If you have close to 1,000 words 
# # (or more), you’ll see many hash collisions, which will decrease the 
# accuracy of this encoding method.
dimensionality <- 1000
max_length <- 10

results <- array(0, dim = c(length(samples), max_length, dimensionality))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    # Use hashFunction::spooky.32() to hash the word into a random integer 
    # index between 0 and 1,000
    index <- abs(spooky.32(words[[i]])) %% dimensionality
    results[[i, j, index]] <- 1
  }
}


# 1.2 - Word embedding ----------------------------------------------------

# Another popular and powerful way to associate a vector with a word is the 
# use of dense word vectors, also called word embeddings. 
# 
# Whereas the vectors obtained through one-hot encoding are binary, sparse 
# (mostly made of zeros), and very high-dimensional (same dimensionality as the 
# number of words in the vocabulary), word embeddings are low-dimensional 
# floating-point vectors (that is, dense vectors, as opposed to sparse vectors);
# 
# Unlike the word vectors obtained via one-hot encoding, word embeddings are l
# earned from data. 
# 
# It’s common to see word embeddings that are 256-dimensional, 512-dimensional, 
# or 1,024-dimensional when dealing with very large vocabularies. 
# 
# On the other hand, one-hot encoding words generally leads to vectors that are 
# 20,000-dimensional or greater (capturing a vocabulary of 20,000 token, in this 
# case).
# So, word embeddings pack more information into far fewer dimensions.
# 
# There are two ways to obtain word embeddings:
#    - Learn word embeddings jointly with the main task you care about (such as
#      document classification or sentiment prediction). In this setup, you 
#      start with random word vectorsand then learn word vectors in the same 
#      way you learn the weights of a neural network.
#    - Load into your model word embeddings that were precomputed using a 
#      different machine-learning task than the one you’re trying to solve. 
#      These are called pretrained word embeddings.


# 1.2.1 - Learning word embeddings with an embedding layer ----------------

## Instantiating an embedding layer

# Two arguments: the number of possible tokens (here, 1,000) and the 
# dimensionality of the embeddings (here, 64).
embedding_layer <- layer_embedding(input_dim = 1000, output_dim = 64)

# An embedding layer takes as input a 2D tensor of integers, of shape (samples,
# sequence_length), where each entry is a sequence of integers. 
#
# It can embed sequences of variable lengths: for instance, you could feed into 
# the embedding layer in the previous example batches with shapes (32, 10) 
# (batch of 32 sequences of length 10) or (64, 15) (batch of 64 sequences of 
# length 15). 
# 
# All sequences in a batch must have the same length, though (because you need 
# to pack them into a single tensor), so sequences that are shorter than others 
# should be padded with zeros, and sequences that are longer should be 
# truncated.
# 
# This layer returns a 3D floating-point tensor, of shape (samples, 
# sequence_length, embedding_dimensionality). Such a 3D tensor can then be
# processed by an RNN layer or a 1D convolution layer.

## Loading the IMDB data for use with an embedding layer

max_features <- 10000
maxlen <- 40

imdb <- dataset_imdb(num_words = max_features)

c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

## Using an embedding layer and classifier on the IMDB data

model <- keras_model_sequential() %>%
  # Specifies the maximum input length to the embedding layer so you can later 
  # flatten the embedded inputs. 
  # After the embedding layer, the activations have shape (samples, maxlen, 8).
  layer_embedding(input_dim = 10000, output_dim = 8,
                  input_length = maxlen) %>%
  # Flattens the 3D tensor of embeddings into a 2D tensor of shape 
  # (samples, maxlen * 8)
  layer_flatten() %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

summary(model)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# 1.2.2 - Using pretrained word embeddings --------------------------------

# code.google.com/archive/p/word2vec
# nlp.stanford.edu/projects/glove

# Raw IMDB - ai.stanford.edu/~amaas/data/sentiment

## Processing the labels of the raw IMDB data

imdb_dir <- here::here("data", "Downloads", "imdb_raw", "aclimdb")
train_dir <- file.path(imdb_dir, "train")

labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
  
  label <- switch(label_type, neg = 0, pos = 1)
  
  dir_name <- file.path(train_dir, label_type)
  
  for (fname in list.files(dir_name, 
                           pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

## Tokenizing the text of the raw IMDB data

# Cuts off reviews after 100 words
maxlen <- 100

# Trains on 200 samples
training_samples <- 200

# Validates on 10,000 samples
validation_samples <- 10000

# Considers only the top 10,000 words in the dataset
max_words <- 10000

tokenizer <- text_tokenizer(num_words = max_words) %>%
  fit_text_tokenizer(texts)

sequences <- texts_to_sequences(tokenizer, texts)

word_index = tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")

data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(labels)

cat("Shape of data tensor:", dim(data), "\n")
cat('Shape of label tensor:', dim(labels), "\n")

# Splits the data into a training set and a validation set. But first shuffles 
# the data,because you’re starting with data in which samples are ordered (all 
# negative first, then all positive).
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]

x_train <- data[training_indices,]
y_train <- labels[training_indices]

x_val <- data[validation_indices,]
y_val <- labels[validation_indices]


## Parsing the GloVe word-embeddings file

glove_dir = here::here("data", "downloads", "glove_6B")

lines <- readLines(file.path(glove_dir, "glove.6B.100d.txt"))

embeddings_index <- new.env(hash = TRUE, parent = emptyenv())

for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
}

cat("Found", length(embeddings_index), "word vectors.\n")

## Preparing the GloVe word-embeddings matrix

# Build an embedding matrix that you can load into an embedding layer. It must 
# be a matrix of shape (max_words, embedding_dim), where each entry i contains
# the embedding_dim-dimensional vector for the word of index i in the reference 
# word index (built during tokenization). Note that index 1 isn’t supposed to 
# stand for any word or token — it’s a placeholder.

embedding_dim <- 100

embedding_matrix <- array(0, c(max_words, embedding_dim))

for (word in names(word_index)) {
  
  index <- word_index[[word]]
  
  if (index < max_words) {
    
    embedding_vector <- embeddings_index[[word]]
    
    # Words not found in the embedding index will be all zeros.
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector
  }
}

## Model definition

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

## Loading pretrained word embeddings into the embedding layer

get_layer(model, index = 1) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

## Training and evaluation

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

save_model_weights_hdf5(model, "pre_trained_glove_model.h5")

## Plotting the results

plot(history)

## Training the same model without pretrained word embeddings

# You can also train the same model without loading the pretrained word 
# embeddings and without freezing the embedding layer. In that case, you’ll be 
# learning a task-specific embedding of the input tokens, which is generally more 
# powerful than pretrained word embeddings when lots of data is available. But in 
# this case, you have only 200 training samples.

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 32,
  validation_data = list(x_val, y_val)
)

plot(history)

## Tokenizing the data of the test set

test_dir <- file.path(imdb_dir, "test")

labels <- c()
texts <- c()

for (label_type in c("neg", "pos")) {
  label <- switch(label_type, neg = 0, pos = 1)
  dir_name <- file.path(test_dir, label_type)
  for (fname in list.files(dir_name, pattern = glob2rx("*.txt"),
                           full.names = TRUE)) {
    texts <- c(texts, readChar(fname, file.info(fname)$size))
    labels <- c(labels, label)
  }
}

sequences <- texts_to_sequences(tokenizer, texts)

x_test <- pad_sequences(sequences, maxlen = maxlen)
y_test <- as.array(labels)
                        
## Evaluating the model on the test set

model %>%
  load_model_weights_hdf5("pre_trained_glove_model.h5") %>%
  evaluate(x_test, y_test)


# 2 - RNNs ----------------------------------------------------------------


# 2.1 - Intro -------------------------------------------------------------

# A recurrent neural network (RNN) processes sequences by iterating through the 
# sequence elements and maintaining a state containing information relative to 
# what it has seen so far.
# an
# RNN is a type of neural network that has an internal loop. The state of the
# RNN is reset between processing two different, independent sequences (such as 
# two different IMDB reviews), so you still consider one sequence a single data 
# point: a single input to the network. What changes is that this data point is 
# no longer processed in a single step; rather, the network internally loops over 
# sequence elements.

# EXAMPLE
# This RNN takes as input a sequence of vectors, which you’ll encode as a 2D 
# tensor of size (timesteps, input_features). 
# It loops over timesteps, and at each timestep, it considers its current state 
# at t and the input at t (of shape (input_features), and combines them to 
# obtain the output at t. 
# You’ll then set the state for the next step to be this  previous output. 
# For the first timestep, the previous output isn’t defined; 
# hence there is no current state. So, you’ll initialize the state as an
# all-zero vector called the initial state of the network.

# PSEUDO CODE
#
# Initital state
# state_t = 0
# 
# # Iterates over sequence elements
# for (input_t in input_sequence) {
#   
#   # output_t <- f(input_t, state_t)
#     output_t <- activation(dot(W, input_t) + dot(U, state_t) + b)
#   
#   # The previous output becomes the new state.
#   state_t <- output_t
# }

## R implementation of a simple RNN

# Number of timesteps in the input sequence
timesteps <- 100
# Dimensionality of the input feature space
input_features <- 32
# Dimensionality of the output feature space
output_features <- 64

random_array <- function(dim) {
  array(runif(prod(dim)), dim = dim)
}

# Input data: random noise for the sake of the example
inputs <- random_array(dim = c(timesteps, input_features))
# Initial state: an all-zero vector
state_t <- rep_len(0, length = c(output_features))

# Creates random weight matrices
W <- random_array(dim = c(output_features, input_features))
U <- random_array(dim = c(output_features, output_features))
b <- random_array(dim = c(output_features, 1))

output_sequence <- array(0, dim = c(timesteps, output_features))

for (i in 1:nrow(inputs)) {
  
  # input_t is a vector of shape (input_features).
  input_t <- inputs[i,]
  
  # Combines the input with the current state (the previous output) to obtain 
  # the current output
  output_t <- tanh(as.numeric((W %*% input_t) + (U %*% state_t) + b))
  
  # Updates the result matrix
  output_sequence[i,] <- as.numeric(output_t)
  
  # Updates the state of the network for the next timestep
  state_t <- output_t
}

# NOTE 
# In this example, the final output is a 2D tensor of shape (timesteps,
# output_features), where each timestep is the output of the loop at time t. 
# Each timestep t in the output tensor contains information about
# timesteps 1 to t in the input sequence—about the entire past. For this
# reason, in many cases, you don’t need this full sequence of outputs; you
# just need the last output (output_t at the end of the loop), because it
# already contains information about the entire sequence.

## A recurrent layer in Keras

# The process you just naively implemented in R corresponds to an actual Keras 
# layer—layer_simple_rnn.
layer_simple_rnn(units = 32)

# There is one minor difference: layer_simple_rnn processes batches of sequences,
# like all other Keras layers, not a single sequence as in the R example. This 
# means it takes inputs of shape (batch_size, timesteps, input_features), rather 
# than (timesteps, input_features).

# Like all recurrent layers in Keras, layer_simple_rnn can be run in two 
# ifferent modes: 
#
#             - The full sequences of successive outputs for each timestep 
#               (a 3D tensor of shape (batch_size, timesteps, output_features))
#
#             - Or only the last output for each input sequence (a 2D tensor of 
#               shape (batch_size, output_features)). 
#
# These two modes are controlled by the return_sequences constructor argument. 
# Let’s look at an example that uses layer_simple_rnn and returns the last 
# state:

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32)

summary(model)

# The following example returns the full state sequence:

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE)

summary(model)

# It’s sometimes useful to stack several recurrent layers one after the other 
# in order to increase the representational power of a network. In such a setup, 
# you have to get all of the intermediate layers to return full sequences:

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 32) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32)

summary(model)


# 2.2 - IMDB example ------------------------------------------------------

## Preparing the IMDB data

max_features <- 10000

maxlen <- 500
batch_size <- 32

cat("Loading data...\n")

imdb <- dataset_imdb(num_words = max_features)

c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb

cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")
cat("Pad sequences (samples x time)\n")

input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)

cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

## Training the model with embedding and simple RNN layers

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

## Plotting results

plot(history)

