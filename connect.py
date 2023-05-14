from tensorflow.keras.utils import img_to_array
import imutils
from keras.models import load_model
import numpy as np
import cv2
from gaze_tracking import GazeTracking
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
from flask import Flask, render_template, url_for, request, jsonify, Response
import PyPDF2
import io
import openai
import xml.etree.ElementTree as ET

openai.api_key = "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"

app = Flask(__name__)

## 고개 각도 감지를 위해 가져옴
def get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val):
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = val[0]
    rear_depth = val[1]
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))
    
    front_size = val[2]
    front_depth = val[3]
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)
    
    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    return point_2d

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix,
                        rear_size=300, rear_depth=0, front_size=500, front_depth=400,
                        color=(255, 255, 0), line_width=2):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    # # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)
    
    
def head_pose_points(img, rotation_vector, translation_vector, camera_matrix):
    rear_size = 1
    rear_depth = 0
    front_size = img.shape[1]
    front_depth = front_size*2
    val = [rear_size, rear_depth, front_size, front_depth]
    point_2d = get_2d_points(img, rotation_vector, translation_vector, camera_matrix, val)
    y = (point_2d[5] + point_2d[8])//2
    x = point_2d[2]
    
    return (x, y)

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

## 감정분석과 동공 움직임 코드가 유사해서 일단 가져옴
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
face_model = get_face_detector()
landmark_model = get_landmark_model()
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

def start():
    while True:
        # We get a new frame from the webcam
        _, frame = webcam.read()
        ##각도움직임
        ret, img = webcam.read()
        size = img.shape

        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        if ret == True:
            faces = find_faces(img, face_model)
            for face in faces:
                marks = detect_marks(img, landmark_model, face)
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
                image_points = np.array([
                                        marks[30],     # Nose tip
                                        marks[8],     # Chin
                                        marks[36],     # Left eye left corner
                                        marks[45],     # Right eye right corne
                                        marks[48],     # Left Mouth corner
                                        marks[54]      # Right mouth corner
                                    ], dtype="double")
                model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])
                camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                            )
                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
                
                # Project a 3D point (0, 0, 1000.0) onto the image plane.
                # We use this to draw a line sticking out of the nose
                
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                
                for p in image_points:
                    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                
                
                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)

                cv2.line(img, p1, p2, (0, 255, 255), 2)
                cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
                # for (x, y) in marks:
                #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
                # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
                try:
                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
                    ang1 = int(math.degrees(math.atan(m)))
                except:
                    ang1 = 90
                    
                try:
                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
                    ang2 = int(math.degrees(math.atan(-1/m)))
                except:
                    ang2 = 90
                    
                    # print('div by zero error')
                if ang1 >= 48:
                    print('Head down')
                    cv2.putText(img, 'Head down', (30, 30), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)
                elif ang1 <= -48:
                    print('Head up')
                    cv2.putText(img, 'Head up', (30, 30), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)
                
                if ang2 >= 48:
                    print('Head right')
                    cv2.putText(img, 'Head right', (90, 30), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)
                elif ang2 <= -48:
                    print('Head left')
                    cv2.putText(img, 'Head left', (90, 30), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)
                
                cv2.putText(frame, str(ang1), tuple(p1), cv2.FONT_HERSHEY_SIMPLEX , 2, (128, 255, 255), 3)
                cv2.putText(frame, str(ang2), tuple(x1), cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 255, 128), 3)
            # cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        if gaze.is_blinking():
            text = "Blinking"
        elif gaze.is_right():
            text = "Looking right"
        elif gaze.is_left():
            text = "Looking left"
        elif gaze.is_center():
            text = "Looking center"

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        
        #frame = webcam.read()[1]
        #reading the frame
        ## 영상 크기가 작아진 거 같어 이게 바뀌나 한번 본다
        #frame = imutils.resize(frame,width=300)
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
                    ##frame으로 바뀜
                    cv2.putText(frame, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                                (0, 0, 255), 2)
                    ##요부분

        cv2.imshow("Demo", frame)
        # cv2.imshow('your_face', frameClone)
        # cv2.imshow("Probabilities", canvas)

        if cv2.waitKey(1) == 27:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result    
# webcam.release()
# cv2.destroyAllWindows()

def makeRequest(messages):
    return openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [messages]
            )

def makeQuestion(article):
    text = ""
    pd_text = []
    pd_text_row = []

    article = article.split(".")

    for text in range(len(article)):
        pd_text_row.append(article[text])

        if (text + 1) % 15 == 0:
            pd_text.append("".join(pd_text_row))
            pd_text_row = []

    result = ""

    print("텍스트를 gpt 녀석에게 요약시키고 있습니다.")
    print("긴 텍스트는 15문장을 기준으로 구분되어 gpt가 기억합니다.")

    #자기소개서 내용 요약. result에 요약한 내용 저장
    for i in range(len(pd_text)):
        currentText = pd_text[i]

        question = {"role":"user", "content": "다음 내용을 읽고 한국말로 요약해줘.\n" + currentText}
        completion = makeRequest(question)
        response = completion['choices'][0]['message']['content'].strip()
        result += response

        print(f"{i + 1}번째 텍스트를 gpt녀석이 기억했습니다.")
    
    question = {"role":"user", "content": "다음 글을 읽고 현재 면접 중이고 너가 면접관이라 생각하고 한국말로 질문을 세가지 해줘.\n" + result}

    print("gpt가 질문을 생성중입니다.")

    completion = makeRequest(question)

    response = completion['choices'][0]['message']['content'].strip()

    return response.split('\n')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/video')
def index():
    return render_template('liveCam2.html')


@app.route('/interview')
def interview():
    return render_template('interview.html')

@app.route('/interviewResult')
def returnList():
    return render_template('interviewResult.html')
    
@app.route('/pdf', methods=['POST'])
def getPdfText():
    # 파일 읽기 시작
    print("파일을 읽고 있습니다")
    
    # js로 요청 넣은 파일을 가져온다.
    file = request.files['file']

    # PyPDF2 라이브러리의 PdfReader 메서드를 통해 파일을 읽는다.
    reader = PyPDF2.PdfReader(file)
    
    article = ''
    
    # 페이지 별로 읽어온 pdf 파일의 텍스트를 article 변수에 더해준다.
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        article += page.extract_text()

    # 만든 전체 자소서 텍스트를 gpt에 넘겨 질문을 생성한다.
    question = makeQuestion(article)

    # 생성된 질문(string type array)을 questionArray라는 key값의 value로 설정한 dictionary를 만든다.
    # 만들어진 dictionary를 jsonify 메서드를 사용해 json형태로 변환한 후 반환해준다.
    return jsonify({"questionArray" : question})

@app.route('/test', methods=['GET'])
def getTest():
    return jsonify({"hi": "hihi"})


if __name__ == '__main__':
    #app.run('127.0.0.1', 5000, debug=True)
    app.run(debug=True)