export const getQuestionFromLocalStorage = () => {
  const question = JSON.parse(localStorage.getItem("question"));

  return question;
};
