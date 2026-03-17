select
    req.problem as problem,
	STRING_AGG(com.comments, ' ||| ' order by com.date_comments) as comments
from request req
left join comments com on req.number = com.number
where req.number = :request_id
group by
    req.number
;