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

  constructor(question, setNextButtonAvailable) {
    this.#question = question;
    this.setNextButtonAvailable = setNextButtonAvailable;
    this.#$questionButton = document.querySelector(".voice");

    this.render();

    this.#$startRecordButton = document.querySelector("#start-record");
    this.#$stopRecordButton = document.querySelector("#stop-record");
    this.#$speechResult = document.querySelector("#speech_result");
    this.#$gptResponse = document.querySelector("#gpt-response");

    this.bindEvents();
  }

  render() {
    this.#$questionButton.insertAdjacentHTML(
      "afterend",
      `
        <span id="speech_result"></span>
        <div id="gpt-response"></div>
        <button type="button" id="start-record" disabled>
            Start Record
        </button>
        <button type="button" id="stop-record" disabled>
            Stop Record
        </button>
    `
    );
  }

  bindEvents() {
    this.#$startRecordButton.onclick = this.startSpeechRecognition.bind(this);
    this.#$stopRecordButton.onclick = this.endSpeechRecognition.bind(this);
  }

  resetResult() {
    this.#$speechResult.innerHTML = "";
    this.#$gptResponse.innerHTML = "";
  }

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

      this.#$speechResult.innerHTML = text;

      const response = await this.getGPTResponse(text);
      console.log(response);
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
    }" 라는 질문에 대한 답변인 "${prompt}"에 대해 네가 면접관이라고 생각하고 0점~10점 중 하나의 점수를 매기고 그 이유를 상세하게 설명해줘`;

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

      this.#$gptResponse.innerHTML = data.choices[0].message.content.replaceAll(
        ".",
        "<br/>"
      );
      this.setNextButtonAvailable();

      return data;
    } catch {
      return "Sorry, I could not generate a response at this time.";
    }
  }
}
