export function renderTable(container, data, options = {}) {

  if (!Array.isArray(data) || data.length === 0) {
    container.innerHTML = "<p>Нет результатов</p>";
    return;
  }

  const keys = Object.keys(data[0]).filter(k => k !== "url");

  const headers = options.headers || {};

  let html = `<table><thead><tr>`;

  keys.forEach(k => {

    html += `<th>${headers[k] || k}</th>`;

  });

  html += `</tr></thead><tbody>`;

  data.forEach(row => {

    html += "<tr>";

    keys.forEach(k => {

      if (k === "id") {

        const link = row.url || "#";

        html += `<td><a href="${link}" target="_blank">${row.id}</a></td>`;

      } else {

        html += `<td>${row[k] ?? ""}</td>`;

      }

    });

    html += "</tr>";

  });

  html += "</tbody></table>";

  container.innerHTML = html;

}