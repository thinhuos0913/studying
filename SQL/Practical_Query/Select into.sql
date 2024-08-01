use HowKteam
go
--Select Into dung cho backup, giup tao ra 1 bang moi dua vao bang da co san va backup du lieu
select * into TableA 
from GIAOVIEN --Lay het du lieu cua bang GIAOVIEN dua vao bang moi ten la TableA, bang nay co cac record tuong duong voi bang GIAOVIEN
select HoTen into TableB
from GIAOVIEN --Tao ra bang TableB moi, co cot du lieu la HoTen tuong ung nhu bang GIAOVIEN
select HOTEN into TableC
from GIAOVIEN
where LUONG>2000
select * from TableC
--Tao 1 bang moi tu nhieu bang
select MAGV, HOTEN, TENBM 
into GVBACKUP
from GIAOVIEN, BOMON
where BOMON.MABM=GIAOVIEN.MABM
select * from GVBACKUP
--Tao ra 1 bang giong nhu GIAOVIEN nhung khong co du lieu
select * into GVBK
from GIAOVIEN
where 0>1 --false
select * from GVBK