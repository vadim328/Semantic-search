import { getProducts, getClients, searchRequests, summarize } from "./api.js";
import { buildPayloadSearch, buildPayloadSum } from "./form.js";
import {
  showLoading,
  hideLoading
} from "./ui.js";
import { renderTable } from "./table.js";
import { initCustomSelect, fillCustomSelect } from "./customSelect.js"; // импорт вспомогательного модуля

// элементы
const productSelect = document.getElementById("product-select");
const clientSelect = document.getElementById("client-select");
const result = document.getElementById("result");
const form = document.getElementById("searchForm");
const summaryForm = document.getElementById("summaryForm");
const summaryResult = document.getElementById("summaryResult");
const btnSearch = document.getElementById("button-search");
const btnSum = document.getElementById("button-sum");

// Инициализация вкладок
function initTabs() {
  const searchTab = document.getElementById("searchTab");
  const summaryTab = document.getElementById("summaryTab");

  const searchSection = document.getElementById("searchSection");
  const summarySection = document.getElementById("summarySection");

  searchTab?.addEventListener("click", () => {
    searchSection.style.display = "block";
    summarySection.style.display = "none";
    searchTab.classList.add("active");
    summaryTab.classList.remove("active");
  });

  summaryTab?.addEventListener("click", () => {
    searchSection.style.display = "none";
    summarySection.style.display = "block";
    summaryTab.classList.add("active");
    searchTab.classList.remove("active");
  });
}

// Основная инициализация
async function init() {
  try {
    const products = await getProducts();
    fillCustomSelect(productSelect, products);  // заполняем кастомный селект
    fillCustomSelect(clientSelect, []);         // пустой клиентский

    initCustomSelect(productSelect);            // навес JS для кастомного селекта
    initCustomSelect(clientSelect);

    // динамическая подгрузка клиентов при выборе продукта
    productSelect.addEventListener("change", async () => {
      const product = productSelect.querySelector(".selected").textContent;
      if (!product || product === "(не выбрано)") {
        fillCustomSelect(clientSelect, []);
        return;
      }
      const clients = await getClients(product);
      fillCustomSelect(clientSelect, clients);
    });

  } catch (e) {
    console.error("Ошибка загрузки продуктов", e);
  }
}

// Поиск
form?.addEventListener("submit", async (e) => {
  e.preventDefault();
  btnSearch.classList.add("onclic");
  showLoading(result);

  try {
    const payload = buildPayloadSearch();
    const data = await searchRequests(payload);

    hideLoading(result);
    renderTable(result, data, { headers: { id: "ID" } });

    btnSearch.classList.remove("onclic");
    btnSearch.classList.add("validate");
    setTimeout(() => btnSearch.classList.remove("validate"), 1500);
  } catch (err) {
    hideLoading(result);
    result.innerHTML = `<p>Ошибка: ${err.message}</p>`;
    btnSearch.classList.remove("onclic");
  }
});

// Суммаризация
summaryForm?.addEventListener("submit", async (e) => {
  e.preventDefault();
  btnSum.classList.add("onclic");
  summaryResult.innerHTML = "Загрузка...";

  try {
    const payload = buildPayloadSum();
    const data = await summarize(payload);
    summaryResult.innerText = data.summary;

    btnSum.classList.remove("onclic");
    btnSum.classList.add("validate");
    setTimeout(() => btnSum.classList.remove("validate"), 1500);
  } catch (err) {
    summaryResult.innerHTML = `<p>Ошибка: ${err.message}</p>`;
    btnSum.classList.remove("onclic");
  }
});

init();
initTabs();