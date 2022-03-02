import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Dense,RepeatVector,LSTM,Dropout,GRU,Dense,SimpleRNN,Embedding 
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
from tensorflow.keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, TimeDistributed
from keras.utils.vis_utils import plot_model
from keras.models import Model, load_model
from keras.layers import Dense, Softmax
from sklearn.metrics import r2_score
from keras.models import Sequential
import matplotlib.pyplot as plt
from pandas import read_csv
from numpy import array
import pandas as pd
import numpy as np
import numpy
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D,Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
!git clone https://github.com/RishitToteja/Chext-X-ray-Images-Data-Set.git
import os

main_dir = "/content/Chext-X-ray-Images-Data-Set/DataSet/Data"

# SETTING TRAIN AND TEST DIRECTORY
train_dir = os.path.join(main_dir, "train")
test_dir = os.path.join(main_dir, "test")

#SETING DIRECTORY FOR COVID AND NORMAL IMAGES DIRECTORY
train_covid_dir = os.path.join(train_dir, "COVID19")
train_normal_dir = os.path.join(train_dir, "NORMAL")

test_covid_dir = os.path.join(test_dir, "COVID19")
test_normal_dir = os.path.join(test_dir, "NORMAL")
# MAKING SEPERATE FILES : 
train_covid_names = os.listdir(train_covid_dir)
train_normal_names = os.listdir(train_normal_dir)

test_covid_names = os.listdir(test_covid_dir)
test_normal_names = os.listdir(test_normal_dir)
import matplotlib.image as mpimg

rows = 4
columns = 4

fig = plt.gcf()
fig.set_size_inches(12,12)

covid_img = [os.path.join(train_covid_dir, filename) for filename in train_covid_names[0:8]]
normal_img = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[0:8]]

print(covid_img)
print(normal_img)

merged_img = covid_img + normal_img

for i, img_path in enumerate(merged_img):
  title = img_path.split("/", 6)[6]
  plot = plt.subplot(rows, columns, i+1)
  plot.axis("Off")
  img = mpimg.imread(img_path)
  plot.set_title(title, fontsize = 11)
  plt.imshow(img, cmap= "gray")

plt.show()
# CREATING TRAINING, TESTING AND VALIDATION BATCHES

dgen_train = ImageDataGenerator(rescale = 1./255,
                                validation_split = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

dgen_validation = ImageDataGenerator(rescale = 1./255,
                                     )

dgen_test = ImageDataGenerator(rescale = 1./255,
                              )

train_generator = dgen_train.flow_from_directory(train_dir,
                                                 target_size = (150, 150), 
                                                 subset = 'training',
                                                 batch_size = 32,
                                                 class_mode = 'binary')
validation_generator = dgen_train.flow_from_directory(train_dir,
                                                      target_size = (150, 150), 
                                                      subset = "validation", 
                                                      batch_size = 32, 
                                                      class_mode = "binary")
test_generator = dgen_test.flow_from_directory(test_dir,
                                               target_size = (150, 150), 
                                               batch_size = 32, 
                                               class_mode = "binary")


print("Class Labels are: ", train_generator.class_indices)
print("Image shape is : ", train_generator.image_shape)
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.applications.xception import Xception
import tensorflow as tf
import keras.backend as K
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())
def sensitivity(y_true, y_pred): 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
def f_beta(y_true, y_pred):
    beta = 1
    precision_value = precision(y_true, y_pred)
    recall_value = sensitivity(y_true, y_pred)
    numer = (1 + beta ** 2) * (precision_value * recall_value)
    denom = ((beta ** 2) * precision_value) + recall_value + K.epsilon()
    return numer / denom
def average_metric(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return  0.5 * (spec + sens)
def conditional_average_metric(y_true, y_pred):
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)

    minimum = K.minimum(spec, sens)
    condition = K.less(minimum, 0.5)

    multiplier = 0.001
    # This is the constant used to substantially lower
    # the final value of the metric and it can be set to any value
    # but it is recommended to be much lower than 0.5

    result_greater = 0.5 * (spec + sens)
    result_lower = multiplier * (spec + sens)
    result = K.switch(condition, result_lower, result_greater)

    return result
def VGG19_Model(image_shape):
  # load the VGG16 network, ensuring the head FC layer sets are left off
  baseModel = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False, input_tensor=tf.keras.layers.Input(shape=image_shape))
  # construct the head of the model that will be placed on top of the the base model
  output = baseModel.output
  output = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(output)
  output = tf.keras.layers.Flatten(name="flatten")(output)
  output = tf.keras.layers.Dense(256, activation="relu")(output)
  output = tf.keras.layers.Dropout(0.5)(output)
  output = tf.keras.layers.Dense(1, activation="sigmoid")(output)
  # place the head FC model on top of the base model (this will become the actual model we will train)
  model = tf.keras.Model(inputs=baseModel.input, outputs=output)
  # loop over all layers in the base model and freeze them so they will not be updated during the first training process
  for layer in baseModel.layers:
    layer.trainable = False
  return model
model=VGG19_Model(train_generator.image_shape)
model.summary()
# COMPILING THE MODEL

model.compile(Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy', conditional_average_metric, specificity, sensitivity])
# TRAINING THE MODEL
history = model.fit(train_generator, 
                    epochs = 50, 
                    validation_data = validation_generator)
# KEYS OF HISTORY OBJECT
history.history.keys()
# PLOT GRAPH BETWEEN TRAINING AND VALIDATION LOSS
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title("Training and validation losses")
plt.xlabel('epoch')
# PLOT GRAPH BETWEEN TRAINING AND VALIDATION ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title("Training and validation accuracy")
plt.xlabel('epoch')
test_generator
test_loss, test_accuracy, test_conditional_average_metric, test_specificity, test_sensitivity = model.evaluate(test_generator)
# GETTING TEST ACCURACY AND LOSS
print("Test Set Loss : ", test_loss)
print("Test Set Accuracy : ", test_accuracy)

from google.colab import files
from keras.preprocessing import image
uploaded = files.upload()
for filename in uploaded.keys():
  img_path = '/content/' + filename
  img = image.load_img(img_path, target_size = (150,150))
  images = image.img_to_array(img)
  images = np.expand_dims(images, axis = 0)
  prediction = model.predict(images)
  if prediction == 0:
    print("The report is COVID-19 Positive")
  else:
    print("The report is COVID-19 Negative")
 model.save("model.h5")
from sklearn.metrics import classification_report, confusion_matrix
y_prob = model.predict_generator(test_generator)
y_pred = np.argmax(y_prob, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['COVID19', 'Normal']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
from sklearn.base import BaseEstimator
from sklearn.metrics import plot_confusion_matrix
class MyEstimator(BaseEstimator):
    def __init__(self):
        self._estimator_type = 'classifier'
        ...
    def predict(self, X):
        return X
estimator = MyEstimator()
plot_confusion_matrix(estimator, y_pred, test_generator.classes)
y_pred
model.summary()
model.save('LSTM.h5')
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# Imports
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ROC Function
def plot_roc_curve(Y_test, model_probs):

	random_probs = [0 for _ in range(len(Y_test))]
	# calculate AUC
	model_auc = roc_auc_score(Y_test, model_probs)
	# summarize score
	print('Model: ROC AUC=%.3f' % (model_auc))
	# calculate ROC Curve
		# For the Random Model
	random_fpr, random_tpr, _ = roc_curve(Y_test, random_probs)
		# For the actual model
	model_fpr, model_tpr, _ = roc_curve(Y_test, model_probs)
	# Plot the roc curve for the model and the random model line
	plt.plot(random_fpr, random_tpr, linestyle='--', label='Random')
	plt.plot(model_fpr, model_tpr, marker='.', label='Model')
	# Create labels for the axis
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# show the legend
	plt.legend()
	# show the plot
	plt.show()
  plot_roc_curve(test_generator.classes, y_prob)
