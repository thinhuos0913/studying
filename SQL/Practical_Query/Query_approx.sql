use HowKteam
go
--Xuất ra thông tin GV mà họ tên bắt đầu bằng chữ "l"
select * from GIAOVIEN
where HOTEN like 'l%'
--Xuất ra thông tin GV mà họ tên kết thúc bằng chữ "n"
select * from GIAOVIEN 
where HOTEN like '%n'
--Xuất ra thông tin GV có họ tên có chữ "n"
select * from GIAOVIEN 
where HOTEN like '%n%'
--Xuất ra thông tin GV có tồn tại chữ "ế"
select * from GIAOVIEN
where HOTEN like N'%ế%'
use HowKteam
go
--Xuat ra thong tin GV co ten bat dau = T va ket thuc = n
select * from GIAOVIEN
where HOTEN like 'T%n'
select * from GIAOVIEN
where HOTEN like '_%g'