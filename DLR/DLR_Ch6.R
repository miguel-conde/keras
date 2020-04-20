library(keras)
library(tidyverse)



# VECTORISING TEXT --------------------------------------------------------

# One-hot encoding --------------------------------------------------------

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


# Word embedding ----------------------------------------------------------

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


# - Learning word embeddings with an embedding layer ----------------------

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


# - Using pretrained word embeddings --------------------------------------

## Processing the labels of the raw IMDB data


