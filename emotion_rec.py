import pandas as pd
import cv2
import numpy as np
dataset_path = 'fer2013/fer2013/fer2013.csv'
image_size=(48,48)

def load_fer2013():
        data = pd.read_csv(dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'),image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        emotions = pd.get_dummies(data['emotion']).as_matrix()
        return faces, emotions

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
# from keras.layers import Activation, Convolution2D, Dropout, Conv2D
# from keras.layers import AveragePooling2D, BatchNormalization
# from keras.layers import GlobalAveragePooling2D
# from keras.models import Sequential
# from keras.layers import Flatten
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import MaxPooling2D
# from keras.layers import SeparableConv2D
# from keras import layers
# from keras.regularizers import l2

# def simple_CNN(input_shape, num_classes):

#     model = Sequential()
#     model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
#                             name='image_array', input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
#     model.add(Dropout(.5))

#     model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
#     model.add(GlobalAveragePooling2D())
#     model.add(Activation('softmax',name='predictions'))
#     return model

# def simpler_CNN(input_shape, num_classes):

#     model = Sequential()
#     model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
#                             name='image_array', input_shape=input_shape))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=16, kernel_size=(5, 5),
#                             strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(.25))

#     model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=32, kernel_size=(5, 5),
#                             strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(.25))

#     model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=64, kernel_size=(3, 3),
#                             strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(.25))

#     model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=128, kernel_size=(3, 3),
#                             strides=(2, 2), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Dropout(.25))

#     model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=128, kernel_size=(3, 3),
#                             strides=(2, 2), padding='same'))

#     model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
#                             strides=(2, 2), padding='same'))

#     model.add(Flatten())
#     #model.add(GlobalAveragePooling2D())
#     model.add(Activation('softmax',name='predictions'))
#     return model

# def tiny_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
#     regularization = l2(l2_regularization)

#     # base
#     img_input = Input(input_shape)
#     x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(5, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     # module 1
#     residual = Conv2D(8, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(8, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(8, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 2
#     residual = Conv2D(16, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 3
#     residual = Conv2D(32, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 4
#     residual = Conv2D(64, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     x = Conv2D(num_classes, (3, 3),
#             #kernel_regularizer=regularization,
#             padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax',name='predictions')(x)

#     model = Model(img_input, output)
#     return model


# def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
#     regularization = l2(l2_regularization)

#     # base
#     img_input = Input(input_shape)
#     x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False)(img_input)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#                                             use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)

#     # module 1
#     residual = Conv2D(16, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(16, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 2
#     residual = Conv2D(32, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(32, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 3
#     residual = Conv2D(64, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(64, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     # module 4
#     residual = Conv2D(128, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(128, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(128, (3, 3), padding='same',
#                         kernel_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     x = Conv2D(num_classes, (3, 3),
#             #kernel_regularizer=regularization,
#             padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax',name='predictions')(x)

#     model = Model(img_input, output)
#     return model

# def big_XCEPTION(input_shape, num_classes):
#     img_input = Input(input_shape)
#     x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
#     x = BatchNormalization(name='block1_conv1_bn')(x)
#     x = Activation('relu', name='block1_conv1_act')(x)
#     x = Conv2D(64, (3, 3), use_bias=False)(x)
#     x = BatchNormalization(name='block1_conv2_bn')(x)
#     x = Activation('relu', name='block1_conv2_act')(x)

#     residual = Conv2D(128, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block2_sepconv1_bn')(x)
#     x = Activation('relu', name='block2_sepconv2_act')(x)
#     x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block2_sepconv2_bn')(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

#     residual = Conv2D(256, (1, 1), strides=(2, 2),
#                       padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = Activation('relu', name='block3_sepconv1_act')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block3_sepconv1_bn')(x)
#     x = Activation('relu', name='block3_sepconv2_act')(x)
#     x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
#     x = BatchNormalization(name='block3_sepconv2_bn')(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])
#     x = Conv2D(num_classes, (3, 3),
#             #kernel_regularizer=regularization,
#             padding='same')(x)
#     x = GlobalAveragePooling2D()(x)
#     output = Activation('softmax',name='predictions')(x)

#     model = Model(img_input, output)
#     return model


# if __name__ == "__main__":
#     input_shape = (64, 64, 1)
#     num_classes = 7
#     #model = tiny_XCEPTION(input_shape, num_classes)
#     #model.summary()
#     #model = mini_XCEPTION(input_shape, num_classes)
#     #model.summary()
#     #model = big_XCEPTION(input_shape, num_classes)
#     #model.summary()
#     model = simple_CNN((48, 48, 1), num_classes)
#     model.summary()
from keras.utils.image_utils import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))
   
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
  
camera = cv2.VideoCapture(0)  

def gen_frames():  
    while True:
        # We get a new frame from the webcam
        _, frame = camera.read()
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
        
        
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
        else: continue

 
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                    w = int(prob * 300)
                    cv2.rectangle(canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 2)
                    cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                                  (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


        # cv2.imshow('your_face', frameClone)
        # cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # camera.release()
    # cv2.destroyAllWindows()
        frame=frameClone
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
@app.route('/')
def index():
        return render_template('liveCam.html')

@app.route('/video_feed')
def video_feed():
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
        app.run('127.0.0.1', 8080, debug=True)