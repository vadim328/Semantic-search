export async function getProducts() {

  const response = await fetch("/search/options/products");
  return await response.json();

}

const clientsCache = new Map();

export async function getClients(product) {

  if (clientsCache.has(product)) {
    return clientsCache.get(product);
  }

  const response = await fetch(
    `/search/options/metadata?product=${encodeURIComponent(product)}`
  );

  const data = await response.json();

  clientsCache.set(product, data.clients);

  return data.clients;
}

export async function searchRequests(payload) {

  const response = await fetch("/search/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return await response.json();

}

export async function summarize(payload) {
  const response = await fetch("/summarization/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return await response.json();
}