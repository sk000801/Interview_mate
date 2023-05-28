// ChatGPT API 키
const API_KEY = "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";

// API 호출 URL
const API_URL = "https://api.openai.com/v1/chat/completions";

// 요청 헤더
const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${API_KEY}`,
};

export class VoiceReader {
  #$questionButton;
  #$startRecordButton;
  #$stopRecordButton;
  #$speechResult;
  #$gptResponse;
  #question;
  recognition = null;
  feedbackList = [];

  constructor(question, setNextButtonAvailable) {
    this.#question = question;
    this.setNextButtonAvailable = setNextButtonAvailable;
    this.#$questionButton = document.querySelector(".voice");
    //아래가 내가 추가한 꼬리질문을 위한 버튼

    this.render();

    this.#$startRecordButton = document.querySelector("#start-record");
    this.#$stopRecordButton = document.querySelector("#stop-record");
    // this.#$speechResult = document.querySelector("#speech-result");
    // this.#$gptResponse = document.querySelector("#gpt-response");

    this.bindEvents();
  }

  // <div id="speechForm" style="font-size: 30px; font-style: italic; font-weigth: bold; color: black">지원자님의 대답: </div>
  // <div id="speech-result" style="color: black"></div>
  // <div id="gptForm" style="font-size: 30px; font-style: italic; font-weigth: bold; color: black">ChatGPT의 평가입니다: </div>
  // <div id="gpt-response" style="color: black"></div>
  render() {
    this.#$questionButton.insertAdjacentHTML(
      "afterend",
      `
        <button type="button" id="start-record" style="color: black; display:flex; align-items: center;" disabled>
            Start Record
        </button>
        <button type="button" id="stop-record" style="color: black; display:flex; align-items: center;" disabled>
            Stop Record
        </button>
    `
    );
  }

  bindEvents() {
    this.#$startRecordButton.onclick = this.startSpeechRecognition.bind(this);
    this.#$stopRecordButton.onclick = this.endSpeechRecognition.bind(this);
  }

  // resetResult() {
  //   this.#$speechResult.innerHTML = "";
  //   this.#$gptResponse.innerHTML = "";
  // }

  setQuestion(question) {
    this.#question = question;
  }

  setRecordAvailable() {
    this.#$startRecordButton.removeAttribute("disabled");
  }

  setRecordDisable() {
    this.#$startRecordButton.setAttribute("disabled", "true");
    this.#$stopRecordButton.setAttribute("disabled", "true");
  }

  checkCompatibility() {
    this.recognition = new (window.SpeechRecognition ||
      window.webkitSpeechRecognition)();
    this.recognition.lang = "ko";
    this.recognition.maxAlternatives = 5;

    if (!this.recognition) {
      alert("You cannot use speech api.");
    }
  }

  startSpeechRecognition() {
    console.log("Start");
    this.setRecordDisable();
    this.#$stopRecordButton.removeAttribute("disabled");

    this.checkCompatibility();

    this.recognition.addEventListener("speechStart", () => {
      console.log("Speech Start");
    });

    this.recognition.addEventListener("speechend", () => {
      console.log("Speech End");
    });

    this.recognition.addEventListener("result", async (event) => {
      const text = event.results[0][0].transcript;
      console.log("Speech Result", event.results);

      // this.#$speechResult.innerHTML = text;

      await this.getGPTResponse(text);
    });

    this.recognition.start();
  }

  endSpeechRecognition() {
    this.recognition.stop();
    this.recognition = null;
    this.setRecordDisable();
  }

  async getGPTResponse(prompt) {
    const question = `"${
      this.#question
    }" 라는 질문에 대한 답변인 "${prompt}"에 대해 네가 면접관이라고 생각하고 0점~10점 중 하나의 점수를 매기고 그 이유를 상세하게 설명해줘. 대답은 점수: , 이유:  이런 형태로 해줘.`;

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: headers,
        body: JSON.stringify({
          model: "gpt-3.5-turbo",
          messages: [{ role: "user", content: question }],
        }),
      });

      const data = await response.json();
      const feedback = await data.choices[0].message.content;
      //.replace(/\./g, "<br/>")

      const INTERVIEW_RESULT_KEY = "interviewResultKey";

      //this.#$gptResponse.innerHTML
      const feedbackList = JSON.parse(
        localStorage.getItem(INTERVIEW_RESULT_KEY)
      );

      // console.log(
      //   `로컬스토리지 저장값:`,
      //   feedbackList,
      //   `새로 추가될 지피티 응답`,
      //   feedback
      // );

      localStorage.setItem(
        INTERVIEW_RESULT_KEY,
        JSON.stringify([...feedbackList, feedback])
      );

      this.setNextButtonAvailable();

      return data;
    } catch {
      return "Sorry, I could not generate a response at this time.";
    }
  }
}
