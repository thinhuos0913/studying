use HowKteam
go
select * from GIAOVIEN
select * from NGUOITHAN
---Union: 2 bảng thống nhất lại với nhau theo cột chung, nếu A không tồn tại trong B thì sẽ thêm vào cho đủ
select MAGV from GIAOVIEN
where LUONG < 2000
union -- lấy các duplicate thì dùng union all
select MAGV from NGUOITHAN
where PHAI=N'Nữ'
---Cách truy vấn tương đương
select GV.MAGV from GIAOVIEN as GV, NGUOITHAN as NT
where GV.LUONG < 2000 
and NT.PHAI=N'Nữ'