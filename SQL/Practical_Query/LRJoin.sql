use HowKteam
go
--Left Join
select * from GIAOVIEN
left join BOMON 
on GIAOVIEN.MABM = BOMON.MABM
select * from GIAOVIEN
inner join BOMON 
on GIAOVIEN.MABM = BOMON.MABM
--Right Join
select * from GIAOVIEN
right join BOMON 
on GIAOVIEN.MABM = BOMON.MABM