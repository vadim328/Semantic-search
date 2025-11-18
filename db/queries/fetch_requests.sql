SELECT
    number,
    problem,
    client,
    c.product,
    registry_date
FROM request req
left join contract c on req.contract=c.id
WHERE registry_date >= :from_date
      and registry_date < :to_date;