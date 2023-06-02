// ChatGPT API 키
const API_KEY = "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";

// API 호출 URL
const API_URL = "https://api.openai.com/v1/chat/completions";

// 요청 헤더
const headers = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${API_KEY}`,
};

// const headers2 = {
//   "Content-Type": "application/json",
//   Authorization: `Bearer ${API_KEY2}`,
// };

export class VoiceReader {
  #$questionButton;
  #$startRecordButton;
  #$stopRecordButton;
  #$speechResult;
  #$gptResponse;
  #question;
  recognition = null;
  feedbackList = [];

  $loaderContainer2;
  $nextButton = document.querySelector("#nextButton");

  constructor(question, setNextButtonAvailable) {
    this.#question = question;
    this.setNextButtonAvailable = setNextButtonAvailable;
    this.#$questionButton = document.querySelector(".voice");
    //아래가 내가 추가한 꼬리질문을 위한 버튼

    this.render();

    this.#$startRecordButton = document.querySelector("#start-record");
    this.#$stopRecordButton = document.querySelector("#stop-record");
    this.$loaderContainer2 = document.querySelector(".loader-container2");

    this.bindEvents();
  }

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

      // console.log(this.$loaderContainer2);
      this.$nextButton.style.display = "none";
      this.$loaderContainer2.removeAttribute("hidden");

      await this.getGPTResponse(text);
    });

    this.recognition.start();
  }

  endSpeechRecognition() {
    this.recognition.stop();
    this.recognition = null;
    this.setRecordDisable();
  }

  // async getGPTGoodResponse(prompt) {
  //   const goodAnswer = `"${
  //     this.#question
  //   }" 라는 질문에 대한 모범 답안을 말해줘.`;

  //   try {
  //     const good_response = await fetch(API_URL, {
  //       method: "POST",
  //       headers: headers2,
  //       body: JSON.stringify({
  //         model: "gpt-3.5-turbo",
  //         messages: [{ role: "user", content: goodAnswer }],
  //       }),
  //     });

  //     const data1 = await good_response.json();
  //     const goodAnswer = await data1.choices[0].message.content;
  //     const GOOD_ANSWER_KEY = "goodAnswerKey";
  //     const goodAnswerList = JSON.parse(localStorage.getItem(GOOD_ANSWER_KEY));

  //     localStorage.setItem(
  //       GOOD_ANSWER_KEY,
  //       JSON.stringify([...goodAnswerList, goodAnswer])
  //     );

  //     return data1;
  //   } catch {
  //     return "Sorry, I could not generate a response at this time.";
  //   }
  // }

  async getGPTResponse(prompt) {
    const question = `"${
      this.#question
    }" 라는 질문에 대한 답변인 "${prompt}"에 대해 네가 면접관이라고 생각하고 0점~10점 중 하나의 점수를 매기고 그 이유를 상세하게 설명해줘. 대답은 점수: , 이유:   형태로 해줘.`;

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: headers,
        body: JSON.stringify({
          model: "gpt-3.5-turbo",
          messages: [{ role: "user", content: question }],
        }),
      });

      const data2 = await response.json();

      const feedback = await data2.choices[0].message.content;

      const INTERVIEW_RESULT_KEY = "interviewResultKey";

      const feedbackList = JSON.parse(
        localStorage.getItem(INTERVIEW_RESULT_KEY)
      );

      localStorage.setItem(
        INTERVIEW_RESULT_KEY,
        JSON.stringify([...feedbackList, feedback])
      );

      this.$nextButton.style.display = "flex";
      this.$loaderContainer2.setAttribute("hidden", "true");
      // form.removeAttribute("hidden");
      this.setNextButtonAvailable();

      return data2;
    } catch {
      return "Sorry, I could not generate a response at this time.";
    }
  }
}
