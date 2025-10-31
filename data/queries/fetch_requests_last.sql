SELECT number, problem
FROM request
WHERE registry_date >= :last_fetch_time
      and registry_date < CURRENT_DATE;