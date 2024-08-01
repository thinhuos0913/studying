use HowKteam
go
--Cấu trúc truy vấn
select * from BOMON --Lấy tất cả dữ liệu trong bảng Bộ Môn
select MABM, TENBM from BOMON --Lấy Mã bộ môn và tên bộ môn trong bảng BOMON
--Đổi tên cột hiển thị
select MABM as '1', TENBM as '2' from BOMON --'1' và '2' là tên muốn đổi
select * from GIAOVIEN
select * from BOMON
--Xuất ra thông tin của từng giáo viên: mã GV + Tên + Tên BM GV đó dạy
select * from GIAOVIEN, BOMON --Gần giống như dạng full join (inner join): kết hợp 2 bảng
select MAGV, HOTEN, TENBM from GIAOVIEN, BOMON --Xuất ra thông tin yêu cầu
select GV.MAGV, GV.HOTEN, BM.TENBM from GIAOVIEN as GV, BOMON as BM -- thêm as giải quyết vấn đề các bảng có chung thuộc tính
--Lấy tất cả mã khoa từ table BOMON
select MAKHOA from BOMON
select distinct MAKHOA from BOMON -- Lấy tất cả mã khoa không trùng nhau từ BOMON
--Đếm số khoa không trùng nhau
select COUNT (distinct MAKHOA) from BOMON
--Lấy tất cả dữ liệu của 10 record đầu tiên trong table BOMON
select top 10 * from BOMON
--Lấy tất cả dữ liệu của 10% đầu tiên trong Table BOMON
select top 10 percent * from BOMON
--Bài tập:
--1. Truy xuất dữ liệu của Table Tham gia đề tài
select * from THAMGIADT
select * from THAMGIADT, GIAOVIEN
select * from GIAOVIEN, KHOA
select MAKHOA, TENKHOA from GIAOVIEN, KHOA
select distinct HOTEN, MAKHOA, TENKHOA from GIAOVIEN, KHOA
select * from THAMGIADT, KHOA, GIAOVIEN
select distinct HOTEN, MAKHOA, TENKHOA from THAMGIADT, KHOA, GIAOVIEN
--2. Lấy ra mã khoa và tên khoa tương ứng
use HowKteam
go
select * from THAMGIADT
select MAKHOA, TENKHOA from KHOA
select MAGV, MADT, STT, MAKHOA, TENKHOA from THAMGIADT, KHOA
--3. Lấy ra MAGV, Tên GV và họ tên người thân tương ứng
select * from GIAOVIEN, NGUOITHAN
select * from NGUOITHAN
select GV.MAGV, HOTEN, TEN from GIAOVIEN as GV, NGUOITHAN where (GV.MAGV=NGUOITHAN.MAGV)
--4. Lấy ra MaGV, TênGV và tên khoa của các GV đó làm việc
select GV.MAGV, GV.HOTEN, K.TENKHOA from GIAOVIEN as GV, BOMON as BM, KHOA as K