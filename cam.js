//dom

const recordButton = document.querySelector(".record-button");
const stopButton = document.querySelector(".stop-button");
const playButton = document.querySelector(".play-button");
const downloadButton = document.querySelector(".download-button");
const previewPlayer = document.querySelector("#preview");
const recordingPlayer = document.querySelector("#recording");

let recorder;
let recordedChunks = [];

//functions
function videoStart() {
  //MediaDevices는 카메라, 마이크 등 현재 연결된 미디어 입력 장치로의 접근 방법을 제공하는 인터페이스
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      previewPlayer.srcObject = stream;
      startRecording(previewPlayer.captureStream());
    });
}

function startRecording(stream) {
  recorder = new MediaRecorder(stream);
  //ondataavailable => 녹화가 시작되면 주기적으로 호출되어 캡쳐한 영상을 얻을 수 있는 이벤트
  recorder.ondataavailable = (e) => {
    recordedChunks.push(e.data);
  };
  recorder.start();
}

function stopRecording() {
  //모든 트랙에서 stop함수를 호출하여 비디오 스트림을 정지하는 함수
  previewPlayer.srcObject.getTracks().forEach((track) => track.stop());
  recorder.stop();
}

function playRecording() {
  //그리고 저 파일이 webm으로만 저장될 수 있어서? 알아서 mp4 파일같은 걸 반환해야할듯
  const recordedBlob = new Blob(recordedChunks, { type: "video/webm" });
  //그냥 다운로드 파일에 저장되던데 잘은 모르겠다..
  recordingPlayer.src = URL.createObjectURL(recordedBlob);
  //기록된 영상을 재생해주는 함수
  recordingPlayer.play();
  downloadButton.href = recordingPlayer.src;
  downloadButton.download = `recording_${new Date()}.webm`;
}

//event
recordButton.addEventListener("click", videoStart);
stopButton.addEventListener("click", stopRecording);
playButton.addEventListener("click", playRecording);
