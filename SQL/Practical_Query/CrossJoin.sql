use CHIANHOM
go
select *
from NHOM_A
cross join NHOM_B
select * 
from NHOM_A
cross join MONTHI
--Cross join voi dieu kien
SELECT * FROM dbo.NHOM_A
CROSS JOIN dbo.MONTHI
WHERE dbo.NHOM_A.TENHS LIKE N'Trần%'