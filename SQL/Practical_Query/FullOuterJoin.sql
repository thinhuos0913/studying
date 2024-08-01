use HowKteam
go
select * from GIAOVIEN, BOMON
where BOMON.MABM=GIAOVIEN.MABM
select * from GIAOVIEN 
full outer join BOMON --Gom 2 bang theo dk, ben nao khong co du lieu thi de null
on BOMON.MABM=GIAOVIEN.MABM 
select * from BOMON
cross join GIAOVIEN
--Cross join la to hop moi record cua bang A voi all record cua bang B, khong can ON
