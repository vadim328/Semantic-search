// Навешивание кликов и открытия/закрытия
export function initCustomSelect(customSelect) {
  const selected = customSelect.querySelector(".selected");
  const options = customSelect.querySelector(".options");

  selected.addEventListener("click", () => {
    document.querySelectorAll(".custom-select").forEach(s => {
      if (s !== customSelect) s.classList.remove("open");
    });
    customSelect.classList.toggle("open");
  });

  document.addEventListener("click", e => {
    if (!e.target.closest(".custom-select")) {
      document.querySelectorAll(".custom-select").forEach(s => s.classList.remove("open"));
    }
  });
}

// Заполнение опциями
export function fillCustomSelect(customSelect, items) {
  const ul = customSelect.querySelector(".options");
  ul.innerHTML = "";

  // добавляем пустую опцию
  const emptyLi = document.createElement("li");
  emptyLi.textContent = "(не выбрано)";
  ul.appendChild(emptyLi);

  emptyLi.addEventListener("click", () => {
    customSelect.querySelector(".selected").textContent = "(не выбрано)";
    customSelect.classList.remove("open");
    customSelect.dispatchEvent(new Event("change"));
  });

  items.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    ul.appendChild(li);

    li.addEventListener("click", () => {
      customSelect.querySelector(".selected").textContent = item;
      customSelect.classList.remove("open");
      customSelect.dispatchEvent(new Event("change"));
    });
  });
}