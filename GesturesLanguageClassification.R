library(crayon)
library(keras)
library(tensorflow)

TR <- read.csv("sign_mnist_train.csv")
TE <- read.csv("sign_mnist_test.csv")

X_train <- TR[,-1] / 255
y_train <- TR[,1]
X_test <- TE[,-1] / 255
y_test <- TE[,1]

X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

n_classes <- length(unique(y_train))
# Sappiamo che ci sono 24 classi perche 'j' e 'z' non fanno parte del dataset dal momento
# che per rapresentarle non basta un immagine ma un video perchè la mano si muove.
# Tuttavia, per codificare le classi con one-hot encoding necessitiamo di passare
# n_classes = 25. Se passassimo 24 avremmo un IndexError poichè non ci sarebbe la colonna
# per la classe 24.
# Con n_classes = 25 risolviamo il problema a costo di avere la colonna 9 -che corrisponde a j-
# sempre vuota, poichè non esistono immagini di questa classe.

y_train <- to_categorical(y_train, n_classes + 1)
y_test <- to_categorical(y_test, n_classes + 1)

# La risoluzione delle immagini del dataset è 28 X 28
img_rows <- 28
img_cols <- 28

png("histTrain.png")
hist(TR$label, xlab = 'Class', col = 'green', main = 'Train Class Distribution')
dev.off()

png("histTest.png")
hist(TE$label, xlab = 'Class', col = 'green', main = 'Test Class Distribution')
dev.off()

build_conv_neural_netwok <- function()
{
  model <- keras_model_sequential() %>%
    layer_reshape(target_shape = c(28, 28, 1), input_shape = c(784)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.5) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = n_classes + 1, activation = "softmax")
  return(model)
}

build_simple_neural_netwok <- function()
{
  model <- keras_model_sequential() %>%
    layer_reshape(target_shape = c(28, 28, 1), input_shape = c(784)) %>%
    layer_flatten() %>%
    layer_dense(units = 512, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 256, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = 128, activation = "relu") %>%
    layer_dropout(rate = 0.25) %>%
    layer_dense(units = n_classes + 1, activation = "softmax")
  return(model)
}

compute_model <- function(model)
{
  # Setting della funzione loss, dell'algoritmo di ottimizzazione e delle metriche di valutazione.
  # Poiche' la funzione di attivazione del layer output e' softmax, che restituisce
  # valori [0,1], allora la funzione di loss più adattta in questo caso e' la 
  # cross entropy.
  model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "SGD",
    metrics = c("accuracy")
  )
  
  batch_size <- 128
  epochs <- 200
  
  # se val_loss non migliora di almeno 0.005 per 5 epoche consecutive, allora il training termina
  earlyStop <- callback_early_stopping(min_delta = 0.005, patience = 5)  
  
  # Training
  history <- model %>% fit(
    X_train, y_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_split = 0.2,
    callbacks = list(earlyStop)
  )
  
  history_df <- as.data.frame(history)
  values <- history_df[history_df$metric == 'accuracy' & history_df$data == 'validation', 'value']
  
  # Valutazione del modello sul dataset di test
  score <- model %>% evaluate(X_test, y_test, verbose = 0)
  
  cat("Test loss:", score[[1]])
  cat(" , Test accuracy:", score[[2]], "\n")
  return(list(values = values, test_accuracy = score[[2]]))
}

CNN_test_accuracy <- 0
NN_test_accuracy <- 0

CNN_val_accuracy <- c()
NN_val_accuracy <- c()


res <- compute_model(build_simple_neural_netwok())
NN_test_accuracy <- res$test_accuracy
NN_val_accuracy <- res$values
NN_val_accuracy <- na.omit(NN_val_accuracy)

res <- compute_model(build_conv_neural_netwok())
CNN_test_accuracy <- res$test_accuracy
CNN_val_accuracy <- res$values
CNN_val_accuracy <- na.omit(CNN_val_accuracy)

cat(red("Simple Neural Network's accuracy on Test Set: ", NN_test_accuracy, '\n'))
cat(red("Convolutional Neural Network's accuracy on Test Set: ", CNN_test_accuracy, '\n'))


png("Accuracy.png")
plot(1:length(NN_val_accuracy), NN_val_accuracy, 
     type = 'b', col = 'green', xlab = 'Epochs',
     ylab = 'Accuracy', pch = 16, main = "Accuracy on the validation set over the epochs")
lines(1:length(CNN_val_accuracy), CNN_val_accuracy, type = 'b', col = 'red', pch = 16)
legend("topleft", 
       legend = c("NN", "CNN"), 
       col = c("green", "red"),
       lty = 1,
       bty = 'n',
       y.intersp = 0.5, 
       bg = NA, border = NA,
       pch = c(16,16),
       )
dev.off()
