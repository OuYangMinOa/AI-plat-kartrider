





# training  file
import tensorflow as tf
from tensorflow.keras import layers, models, Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, AveragePooling2D, GlobalAveragePooling2D, Softmax, Dropout
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet   import ResNet101, ResNet50, ResNet152
from ModifiedTensorBoard import ModifiedTensorBoard
import numpy as np
import os, cv2

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)



def model_1(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.Adam(0.01),loss="mse",
    metrics=["accuracy"])
    return model

def model_2(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(127, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(63, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(27, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.Adam(),loss="mse",
    metrics=["accuracy"])
    return model

def model_3(input_size):
    base = Xception(weights='imagenet', input_shape = (input_size))
    x = base.output
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(6, activation='relu')(x)
    x = layers.Softmax()(x)
    model = Model(inputs = base.input,outputs = x)
    model.compile(loss = "mse" , metrics = ["accuracy"], optimizer = tf.optimizers.Adam() )
    #model.summary()
    return model


def model_4(input_size):
    base = ResNet101(weights=None, input_shape = (input_size))
    x = base.output
    x = layers.Dense(1024, activation='tanh')(x)
    x = layers.Dense(512, activation='tanh')(x)
    x = layers.Dense(6, activation='tanh')(x)
    x = layers.Softmax()(x)
    model = Model(inputs = base.input,outputs = x)
    model.compile(loss = "CategoricalCrossentropy" , metrics = ["accuracy"], optimizer = tf.optimizers.SGD(learning_rate=0.000002) )
    #model.summary()
    return model


def model_5(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1025, activation='tanh'))
    model.add(layers.Dense(521, activation='tanh'))
    model.add(layers.Dense(6, activation='tanh'))
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.2),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model

def model_6(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='tanh', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='tanh'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1025, activation='tanh'))
    model.add(layers.Dense(521, activation='tanh'))
    model.add(layers.Dense(6, activation='tanh'))
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.2),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model

def model_7(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1025, activation='tanh'))
    model.add(layers.Dense(521, activation='tanh'))
    model.add(layers.Dense(6, activation='linear'))
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.02),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model


def model_8(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))   
    model.add(layers.Dense(521, activation='tanh'))
    model.add(Dropout(0.5))   
    model.add(layers.Dense(6, activation='tanh'))   
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.02),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model


def model_9(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(6, activation='tanh'))   
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.00002),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model


def model_10(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(6, activation='tanh'))   
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.000002),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model

def model_11(input_size):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(input_size)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(612, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(layers.Dense(32, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(6, activation='tanh'))   
    model.add(layers.Softmax())
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.000002),loss="CategoricalCrossentropy",
    metrics=["accuracy"])
    return model



RECORD_WIDTH = 240
RECORD_HEIGHT = 160
IS_RGB = True

DATAFLODER = "processed_data" # "dataset"
NOW_MODEL = model_11
MODEL_FLODER = "models"
MODEL_NUM = "model_11.h5"
MODEL_NAME = f"{MODEL_FLODER}\\{MODEL_NUM}"

if (os.path.isfile(MODEL_NAME)):
    print(f"Load exist model {MODEL_NAME} ...")
    m = models.load_model(MODEL_NAME)
else:
    print(" No found model ")   
    print(f"Creating new model : {MODEL_NAME} ...")
    m = NOW_MODEL((RECORD_HEIGHT,RECORD_WIDTH,3 if (IS_RGB)else 1))

train_data = [f"{DATAFLODER}\\{file}" for file in os.listdir(DATAFLODER)]
DATA_NUM = len(train_data)


print(f"\nGoing to train in {DATA_NUM} datas")
tensorboard = ModifiedTensorBoard(log_dir = f"logs/{MODEL_NUM}")
# tensorboard = tf.keras.callbacks.TensorBoard(f"logs/{MODEL_NUM}")

for EPOCH in range(10000):
    print(f"\n ==== {EPOCH+1} ====\n")
    # np.random.shuffle(train_data)

    for num, file in enumerate(train_data):
        FILENAME = file
        print(f"{num+1} \\ {DATA_NUM}")
        print(FILENAME)
        
        print("Reading data...")
        data = np.load(FILENAME,allow_pickle=True)
        print("Data loaded")    

        if (data[0].shape != ((RECORD_WIDTH,RECORD_HEIGHT))):
            print("reshaping the data ...")
            x = np.array([cv2.resize(each[0],(RECORD_WIDTH,RECORD_HEIGHT)) for each in list(data)])/255.0
            y = np.array([each[1] for each in list(data)])

        else:
            x = np.array([each[0] for each in list(data)])/255.0
            y = np.array([each[1] for each in list(data)])

        # shuffle_index = np.arange(len(x))
        # np.random.shuffle(shuffle_index)


        # # print(shuffle_index)
        # x = x[shuffle_index]
        # y = y[shuffle_index]

        if (not IS_RGB):
            x = x.reshape((-1,*x.shape,1))[0]

        m.fit(x,y,epochs = 1,batch_size=4)
        m.save(MODEL_NAME)
        tf.keras.backend.clear_session()
