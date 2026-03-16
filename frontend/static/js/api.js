export async function getProducts() {

  const res = await fetch("/search/options/products");
  return await res.json();

}

const clientsCache = new Map();

export async function getClients(product) {

  if (clientsCache.has(product)) {
    return clientsCache.get(product);
  }

  const res = await fetch(
    `/search/options/metadata?product=${encodeURIComponent(product)}`
  );

  const data = await res.json();

  clientsCache.set(product, data.clients);

  return data.clients;
}

export async function searchRequests(payload) {

  const res = await fetch("/search/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  return await res.json();

}