library(tidyverse)

# Array definition
my_array = array(data = rnorm(10*24*4),
                 dim = c(10, 24, 4),
                 dimnames = list(n_day = 1:10,
                                 hour = 1:24,
                                 price = c("Open", "Close", "Max", "Min")))
dim(my_array)
dimnames(my_array)

# rank, shape, type
my_rank <- length(dim(my_array))
names(dim(my_array))
my_shape <- dim(my_array)
my_type = typeof(my_array)

# Arrays slicing
my_array[1:3, ,]
my_array[1:3, 15:22,]
my_array[1:3, 15:22, c("Open", "Close")]

# reshaping (array_reshape)



# Bitwise operations (+, -, *, /, pmax() = softmax, pmin() = softmin)

# apply
my_array[1:3, 15:22, c("Open", "Close")] %>% apply(c("n_day", "price"), cumsum)

cum_array <- my_array %>% apply(c("n_day", "price"), cumsum)

# sweep


# aperm (Generalization of t())
cum_array %>% aperm(c(2, 1, 3))
cum_array <- cum_array %>% aperm(names(dimnames(my_array)))
