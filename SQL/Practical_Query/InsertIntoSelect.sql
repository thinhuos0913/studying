use HowKteam
go
select * into CloneGV
from GIAOVIEN
where 1=0 --false
insert into CloneGV --copy du lieu vao bang da ton tai
select * from GIAOVIEN
select * from CloneGV
select * from GIAOVIEN