SELECT number, problem, registry_date
FROM request
WHERE registry_date < :last_fetch_time;