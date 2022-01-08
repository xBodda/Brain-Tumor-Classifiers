import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from skimage.io import imread 
import keras 
from keras import Sequential 
from tensorflow.keras.applications.mobilenet import MobileNet
from keras.layers import Dense 
from keras.preprocessing import image 
import tensorflow as tf 
import tensorflow.keras.layers as layers 
import warnings 
from random import shuffle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
from sklearn.metrics import confusion_matrix , classification_report

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

brain_df = pd.read_csv('../Dataset/BrainTumor.csv', usecols=[0,1])
brain_df.head()


brain_df.isnull().sum()


brain_df['Class'].value_counts()


sns.countplot(brain_df['Class'])



path_list = []
base_path = '../Images'
for entry in os.listdir(base_path):
    path_list.append(os.path.join(base_path, entry))

    
paths_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in path_list}
brain_df['Path'] = brain_df['Image'].map(paths_dict.get)
brain_df.head()


for x in range(0,9):
    plt.subplot(3,3,x+1)
    
    plt.xticks([])
    plt.yticks([])
    img = imread(brain_df['Path'][x])
    plt.imshow(img)
    plt.xlabel(brain_df['Class'][x])

    
brain_df['split'] = np.random.randn(brain_df.shape[0], 1)

msk = np.random.rand(len(brain_df)) <= 0.8

train_df = brain_df[msk]
test_df = brain_df[~msk]
train_df.to_csv('brain_tumor_train.csv', index=False)
test_df.to_csv('brain_tumor_test.csv', index=False)
train_list = train_df.values.tolist()
test_list = test_df.values.tolist()

def generator(samples, batch_size=32,shuffle_data=True):
    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    while True: 
        shuffle(samples)

       
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

           
            X_train = []
            y_train = []

           
            for batch_sample in batch_samples:
                label = batch_sample[1]
                img_path = batch_sample[2]
                img =  cv2.imread(img_path)
                
                
                img = img.astype(np.float32)
                X_train.append(keras.applications.nasnet.preprocess_input(img))
                y_train.append(label)

            
            X_train = np.array(X_train)
            y_train = np.array(y_train)

                     
            yield X_train, y_train

           
train_generator = generator(train_list)
test_generator = generator(test_list)



model = Sequential([
    
    MobileNet(input_shape=(224, 224, 3),include_top=False, weights='imagenet'),
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(units=1, activation='sigmoid',name='preds'),   
])
model.layers[0].trainable= False

model.summary()


model.compile(
    
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
    metrics=['binary_accuracy']
)


batch_size = 32
train_size = len(train_list)
test_size = len(test_list)
steps_per_epoch = train_size//batch_size
validation_steps = test_size//batch_size

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)


history = model.fit_generator(
    train_generator,
    steps_per_epoch = steps_per_epoch,
    epochs=5,
    validation_data=test_generator,
    validation_steps = validation_steps,
    verbose=1,
    callbacks = [early_stopping]
)
model.save("model_brain_adam.h5")
print("Saved model to disk")



history_df = pd.DataFrame(history.history)
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))


              
pretrained_cnn = keras.models.load_model('./model_brain_adam.h5')
eval_score = pretrained_cnn.evaluate(test_generator, steps = validation_steps)
print('Eval loss:',eval_score[0])
print('Eval accuracy:',eval_score[1])


y_pred = np.rint(pretrained_cnn.predict_generator(test_generator, steps = validation_steps)).astype(int)
y_test = [i[1] for i in test_list[0:-2]]
target_classes = ['No Tumor','Tumor']
y_test = y_test[0:706]

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
plt.show()
