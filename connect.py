from flask import Flask, render_template, url_for, request, jsonify, Response
import PyPDF2
import io
import openai
import xml.etree.ElementTree as ET
import gtts
from playsound import playsound

openai.api_key = "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"

app = Flask(__name__)

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
    response.split('\n')
    tts = gtts.gTTS(response, lang = "kr")
    tts.save("fortest.mp3")
    playsound("fortest.mp3")

    return response.split('\n')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/interview')
def interview():
    return render_template('interview.html')
    
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