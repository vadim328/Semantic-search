export function buildPayloadSearch() {

  const payload = {
    query: document.getElementById("query").value,
    product: document.getElementById("product").value
  };

  const limit = document.getElementById("limit").value;
  if (limit) payload.limit = Number(limit);

  const alpha = document.getElementById("alpha").value;
  if (alpha) payload.alpha = Number(alpha);

  payload.exact = !document.getElementById("exact").checked;

  const filter = {};

  ["date_from", "date_to"].forEach(id => {

    const val = document.getElementById(id).value;
    if (val) filter[id] = val;

  });

  const client = document.getElementById("client").value;
  if (client) filter.client = client;

  if (Object.keys(filter).length) {
    payload.filter = filter;
  }

  return payload;

}

export function buildPayloadSum() {
    const payload = {
        text: document.getElementById("summaryText").value,
    };
}