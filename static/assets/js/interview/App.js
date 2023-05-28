import { getQuestionFromLocalStorage } from "./domain/index.js";
import { g_gout } from "./utils/voice.js";
import { VoiceReader } from "./voiceReading/index.js";

const $interviewInput = document.querySelector("#interview");
const $interviewButton = document.querySelector(".voice");
const $nextButton = document.querySelector("#nextButton");
const $subTitle = document.querySelector("#question-subtitle");

const INTERVIEW_RESULT_KEY = "interviewResultKey";

export class App {
  #voiceReader;
  #question;
  #index = 0;

  constructor() {
    this.#question = getQuestionFromLocalStorage();

    const currentQuestion = this.#question[this.#index++];

    this.#voiceReader = new VoiceReader(
      currentQuestion,
      this.setNextButtonAvailable.bind(this)
    );
    $interviewInput.value = currentQuestion;

    this.bindEvents();
  }

  sendAnswerFeedback() {
    window.location.href = `/interviewResult`;
  }

  setNextButtonAvailable() {
    $nextButton.removeAttribute("disabled");
  }

  setNextButtonDisable() {
    $nextButton.setAttribute("disabled", "true");
  }

  bindEvents() {
    $interviewButton.onclick = () => {
      g_gout(`interview`);

      this.#voiceReader.setRecordAvailable();
    };

    $nextButton.onclick = () => {
      const currentQuestion = this.#question[this.#index++];

      if (!currentQuestion) {
        alert("종료!");

        this.#voiceReader.setRecordDisable();
        this.setNextButtonDisable();
        //this.#voiceReader.resetResult();

        $subTitle.innerHTML = ``;

        fetch("/stop-video", { method: "PATCH" });

        this.sendAnswerFeedback();
        //this.#voiceReader.feedbackList = [];

        return;
      }

      this.setNextButtonDisable();
      this.#voiceReader.setRecordDisable();

      $subTitle.innerHTML = `질문 ${this.#index}`;

      $interviewInput.value = currentQuestion;
      this.#voiceReader.setQuestion(currentQuestion);
      //this.#voiceReader.resetResult();
      // this.#voiceReader.feedbackList = [];
    };
  }
}
