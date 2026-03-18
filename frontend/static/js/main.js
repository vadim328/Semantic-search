import { getProducts, getClients, searchRequests, summarize } from "./api.js";
import { buildPayloadSearch, buildPayloadSum } from "./form.js";
import {
  fillSelect,
  renderClients,
  showLoading,
  hideLoading
} from "./ui.js";
import { renderTable } from "./table.js";

const productSelect = document.getElementById("product");
const clientSelect = document.getElementById("client");
const result = document.getElementById("result");
const form = document.getElementById("searchForm");

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

async function init() {

  try {

    const products = await getProducts();
    fillSelect(productSelect, products);

  } catch (e) {

    console.error("Ошибка загрузки продуктов", e);

  }

}

productSelect.addEventListener("change", async () => {

  const product = productSelect.value;

  if (!product) return;

  try {

    const clients = await getClients(product);
    renderClients(clientSelect, clients);

  } catch (e) {

    console.error("Ошибка загрузки клиентов", e);

  }

});

form?.addEventListener("submit", async (e) => {

  e.preventDefault();

  showLoading(result);

  try {

    const payload = buildPayloadSearch();
    const data = await searchRequests(payload);

    hideLoading(result);

    renderTable(result, data, {
      headers: {
        id: "ID"
      }
    });

  } catch (err) {

    hideLoading(result);
    result.innerHTML = `<p>Ошибка: ${err.message}</p>`;

  }

});

const summaryForm = document.getElementById("summaryForm");
const summaryResult = document.getElementById("summaryResult");

summaryForm?.addEventListener("submit", async (e) => {
  e.preventDefault();

  summaryResult.innerHTML = "Загрузка...";

  try {
    const payload = buildPayloadSum();

    const data = await summarize(payload);

    summaryResult.innerText = data.summary;

  } catch (err) {
    summaryResult.innerHTML = `<p>Ошибка: ${err.message}</p>`;
  }
});

init();
initTabs();