export function buildPayloadSearch() {

  const payload = {
    query: document.getElementById("query").value,
  };

  const productSelect = document.getElementById("product-select");
  const selectedProduct = productSelect?.querySelector(".selected")?.textContent || "";
  if (selectedProduct && selectedProduct !== "(не выбрано)") {
    payload.product = selectedProduct;
  } else {
    throw new Error("Продукт обязателен"); // если нужно, чтобы был обязательный
  }

  const limit = document.getElementById("limit").value;
  if (limit) payload.limit = Number(limit);

  const alpha = document.getElementById("alpha").value;
  if (alpha) payload.alpha = Number(alpha);

  // Режим поиска (Full | Base | Comments)
  const selectedMode = document.querySelector('input[name="mode"]:checked').value;
  payload.mode = selectedMode;

  payload.exact = document.getElementById("exact").checked;

  const filter = {};

  const clientSelect = document.getElementById("client-select");
  const selectedClient = clientSelect?.querySelector(".selected")?.textContent;
  if (selectedClient && selectedClient !== "(не выбрано)") {
    filter.client = selectedClient;
  }

  ["date_from", "date_to"].forEach(id => {

    const val = document.getElementById(id).value;
    if (val) filter[id] = val;

  });

  if (Object.keys(filter).length) {
    payload.filter = filter;
  }

  return payload;

}

export function buildPayloadSum() {
    const payload = {
        text: document.getElementById("summaryText").value,
    };

    return payload;
}