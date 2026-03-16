import { getProducts, getClients, searchRequests } from "./api.js";
import { buildPayload } from "./form.js";
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

form.addEventListener("submit", async (e) => {

  e.preventDefault();

  showLoading(result);

  try {

    const payload = buildPayload();
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

init();