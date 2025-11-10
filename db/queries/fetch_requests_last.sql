SELECT number, problem, registry_date
FROM request
WHERE registry_date >= :last_fetch_time
      and registry_date < '2025-11-11';