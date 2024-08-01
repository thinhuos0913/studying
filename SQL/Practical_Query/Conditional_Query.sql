--Truy vấn có điều kiện
use HowKteam
go
select GV.MAGV, HOTEN, TEN from GIAOVIEN as GV, NGUOITHAN as NT where (GV.MAGV=NT.MAGV)
--Lấy ra GV lương lớn hơn 2000
select * from GIAOVIEN where LUONG > 2000
--Lấy ra GV là nữ và lương lớn hơn 2000
select * from GIAOVIEN where LUONG > 2000 and PHAI=N'Nữ'
--Lấy ra GV lớn hơn 40 tuổi
select * from GIAOVIEN where YEAR(getdate()) - YEAR(NGSINH) > 40
--Lấy ra họ tên GV, năm sinh và tuổi của GV nhỏ hơn hoặc bằng 40 tuổi
select HOTEN, NGSINH, YEAR(getdate()) - YEAR(NGSINH) from GIAOVIEN where YEAR(getdate()) - YEAR(NGSINH) <= 40
--Giải BT 4
select GV.MAGV, GV.HOTEN, K.TENKHOA from GIAOVIEN as GV, BOMON as BM, KHOA as K
where GV.MABM=BM.MABM and BM.MAKHOA=K.MAKHOA
--Lấy ra GV là trưởng BM
select GV.* from GIAOVIEN as GV, BOMON as BM
where GV.MAGV=BM.TRUONGBM
--Đếm số lượng GV: hàm COUNT
select COUNT(MAGV) as N'Số lượng GV' from GIAOVIEN
--Đếm số lượng người thân của GV có mã GV là 007
select COUNT (*) as N'Số lượng NT' from GIAOVIEN as GV, NGUOITHAN as NT
where GV.MAGV='007' and GV.MAGV=NT.MAGV
select * from GIAOVIEN as GV, NGUOITHAN as NT
where GV.MAGV='007' 
and GV.MAGV=NT.MAGV
--Lấy ra tên GV và tên đề tài người đó tham gia
select HOTEN, TENDT from GIAOVIEN as GV, DETAI as DT, THAMGIADT as TGDT
where GV.MAGV=TGDT.MAGV
and DT.MADT=TGDT.MADT
--Xuất ra mức lương trung bình của các GV (dùng hàm AVG() để tính tb)
use HowKteam
go
select AVG(LUONG)as N'Mức lương TB' from GIAOVIEN
--Xuất ra lương trung bình của GV nữ
select AVG(LUONG)as N'Lương TB của GV nữ' from GIAOVIEN where PHAI=N'Nữ'
--Lấy ra tên GV và tên đề tài mà GV đó tham gia nhiều hơn 1 lần? dùng truy vấn lồng
--Xuất ra tổng kinh phí dành cho các đề tài (dùng hàm SUM() để tính tổng)
select SUM(KINHPHI) as 'Tổng kinh phí' from DETAI
--Xuất ra tổng kinh phí dành cho các đề tài kết thúc trước năm 2009
select SUM(KINHPHI) as 'Tổng kinh phí trước năm 2009' from DETAI where YEAR(NGAYKT)<2009
--Xuất ra tổng lương của GV Nam có năm sinh trước 1960
select SUM(LUONG) as 'Tổng lương của GV Nam sinh trước 1960'
from GIAOVIEN 
where PHAI=N'Nam' and YEAR(NGSINH)<1960
select LUONG, PHAI, NGSINH
from GIAOVIEN 
where PHAI=N'Nam' and YEAR(NGSINH)<1960
--Bài tập
--1. Xuất ra thông tin của Giáo viên và GVQLCN của người đó
select * from GIAOVIEN
select GV1.HOTEN, GV2.HOTEN as 'GV QUẢN LÝ' 
from GIAOVIEN as GV1, GIAOVIEN as GV2 
where GV1.GVQLCM=GV2.MAGV 
--2. Xuất ra số lượng GV của Khoa CNTT
select * from GIAOVIEN
select COUNT(MAGV) as 'Số lượng GV khoa CNTT' 
from KHOA, GIAOVIEN 
where MAKHOA='CNTT'
select COUNT(*) from GIAOVIEN as GV, BOMON as BM
where GV.MABM=BM.MABM and BM.MAKHOA='CNTT'
--3. Xuất ra thông tin GV và đề tài người đó tham gia khi kq là đạt
use HowKteam
go
select GV.MAGV, GV.HOTEN, DT.TENDT, TGDT.KETQUA
from THAMGIADT as TGDT, GIAOVIEN as GV, DETAI as DT
where KETQUA=N'Đạt' and GV.MAGV=TGDT.MAGV
select GV.*
from GIAOVIEN as GV, THAMGIADT as TG
where GV.MAGV=TG.MAGV and KETQUA=N'Đạt'
