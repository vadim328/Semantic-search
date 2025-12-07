SELECT servicecall, emp.fio
FROM request req
left join employee emp on req.resp_user=emp.id
WHERE number = ANY(:numbers)