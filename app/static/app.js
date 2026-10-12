// Общие обработчики форм.
document.addEventListener("submit", function (event) {
  var form = event.target;
  // Подтверждение опасных действий: текст берётся из data-confirm и не попадает в код JavaScript.
  var message = form.getAttribute("data-confirm");
  if (message && !window.confirm(message)) {
    event.preventDefault();
    return;
  }
  // Отметка клиники, для которой открыта страница: если в другой вкладке уже выбрали другую клинику,
  // сервер не выполнит действие и предупредит.
  var clinic = document.body.getAttribute("data-clinic");
  if (clinic && form.method.toLowerCase() === "post") {
    var url = new URL(form.getAttribute("action") || window.location.href, window.location.href);
    url.searchParams.set("_clinic", clinic);
    form.setAttribute("action", url.pathname + url.search);
  }
});
