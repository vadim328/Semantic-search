SELECT servicecall, emp.fio, admission_prority
FROM request req
left join employee emp on req.resp_user=emp.id
WHERE number = ANY(:numbers)
ORDER BY array_position(:numbers, req.number)