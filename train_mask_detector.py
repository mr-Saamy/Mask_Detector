# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import Libraries

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# %% [markdown]
# ## Initialize Initial learning rate, no. of epochs to train for, batch size

# %%
INIT_LR = 1e-4
EPOCHS= 20
BatSiz = 32

# %% [markdown]
# ## Grab the list of images in the dataset and initialize the list of data

# %%
directory = os.path.expanduser("~")+"/Maskk/masks"
cat = ["1","0"]
print('[INFO] Images are loading...')
data=[]
labels=[]


# %%
for category in cat:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        labels.append(category)

# %% [markdown]
# ## perform one-hot encoding of labels

# %%
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)
(trainX, testX,trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# %% [markdown]
# ## construct training image generator for data augmentation

# %%
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# %% [markdown]
# ## load the MobileNetv2 
# ## ensure FC layer sets are left off

# %%
baseModel = MobileNetV2(weights = "imagenet", 
                        include_top=False,
                        input_tensor=Input(shape=(224,224,3)))

# %% [markdown]
# ## head of the model that will be placed on top of the base model(actual training model)

# %%
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)##prevent overfitting
headModel = Dense(2, activation="softmax")(headModel)

# %% [markdown]
# ## place the head FC model on top of the base model 
# 

# %%
model = Model(inputs=baseModel.input, outputs=headModel)

# %% [markdown]
# ## loop over all the layers in the base model and freeze them
# ## this ensures that they will not update during the first training process

# %%
for layer in baseModel.layers:
    layer.trainable = False

# %% [markdown]
# ## Compiling the model

# %%
print('[INFO] compiling the model...')
opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# %% [markdown]
# ## Train the head of the network

# %%
print("[INFO] Training head...")
H = model.fit(aug.flow(trainX,trainY,batch_size=BatSiz),
             steps_per_epoch=len(trainX)//BatSiz,
             validation_data=(testX,testY),
             validation_steps=len(testX)//BatSiz,
             epochs=EPOCHS)

# %% [markdown]
# ## make predictions on the testing set

# %%
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BatSiz)

# %% [markdown]
# ## for each image in the testing set we need to find the index of the
# ## label with corresponding largest predicted probability

# %%
predIdxs = np.argmax(predIdxs, axis=1)

# %% [markdown]
# ## show a nicely formatted classification report

# %%
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# %% [markdown]
# ## serialize the model to disk

# %%
print("[INFO] saving mask detector model...")
model.save("mask_detector.model",save_format="h5")

# %% [markdown]
# ## plot the training loss and accuracy

# %%
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["accuracy"], label="train_accuracy")
#plt.plot(np.arange(0,N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")
print(H.history.keys())


# %%



