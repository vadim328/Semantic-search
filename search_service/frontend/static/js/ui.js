export function fillSelect(select, values) {

  values.forEach(v => {

    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;

    select.appendChild(opt);

  });

}

export function renderClients(select, clients) {

  select.innerHTML = '<option value="">(не выбрано)</option>';

  fillSelect(select, clients);

}

export function showLoading(container) {

  container.innerHTML = "";
  container.classList.add("loading");

}

export function hideLoading(container) {

  container.classList.remove("loading");

}