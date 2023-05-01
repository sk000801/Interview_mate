// pdf-send-form 이란 클래스를 가진 form태그를 가져옴
const form = document.querySelector(".pdf-send-form");
const input = form.querySelector("input");

// 가져온 form에서 submit 이벤트가 발생할 때 다음과 같은 과정을 거친다.
form.addEventListener("submit", (event) => {
  event.preventDefault();

  // 1. input 태그에서 선택된 파일을 가져온다.
  const file = input.files[0];

  // 2. 새로운 FormData 객체를 생성한다.
  const formData = new FormData();

  // 3. 생성한 FormData에 1에서 가져온 파일을 추가해준다.
  formData.append("file", file);

  // 4. fetch api를 사용해 flask 서버의 '/pdf' 경로의 api에 요청을 넣는다.
  //async-await 처리를 통해 response가 오면 아래의 과정이 처리 되도록 한다.
  const response = fetch("/pdf", {
    method: "POST",
    body: formData,
  })
    // 5. 요청의 반환값을 json형태로 변환한다.
    //값이 반환될 때는 Response로 들어오는데 이 안의 data를 가져와야 하므로 json메서드를 호출한다.
    .then((response) => response.json())
    .then((createdQuestion) => {
      console.log(createdQuestion["questionArray"]);
      // 6. 이제 가져온 녀석 처리하면 됩니당.
      localStorage.setItem(
        "question",
        // 문장에 쉼표가 포함될 경우 문제가 발생할 수 있음.
        JSON.stringify(
          createdQuestion["questionArray"]
            .replaceAll(",", "")
            .replaceAll("\n", ",")
            .split(",")
        )
        // .filter((s) => !/\d./.test(s))
      );

      window.location.href = "/video";
    })
    .catch((error) => console.log(error.message));
});
