SELECT
    number,
    problem,
    client,
    c.product,
    registry_date
FROM request req
left join contract c on req.contract=c.id
WHERE registry_date >= :last_fetch_time
    AND registry_date < '2025-11-14';