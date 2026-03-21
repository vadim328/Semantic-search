select
	req.number,
	req.client,
    c.product,
    req.registry_date,
    req.date_end,
    req.problem as problem,
	STRING_AGG(com.comments, ' ||| ' order by com.date_comments) as comments
from request req
left join comments com on req.number = com.number
left join contract c on req.contract=c.id
where req.date_end >= :from_date
	and req.date_end < :to_date
	and (c.product = 'Naumen Erudite' or c.product = 'NCC')
group by
    req.number,
    req.problem,
    req.client,
    c.product;