use HowKteam
go
--Ví dụ 1: kiểm tra xem GV 001 có phải là GVQLCM hay không?
--Cần lấy ra danh sách các mã GV QLCM
--Kiểm tra MAGV tồn tại trong danh sách đó
select * from GIAOVIEN
where MAGV='001' --Truy vấn lồng trong where
/*kiểm tra xem 001 tồn tại trong danh sách*/
and MAGV in
( 
 select GVQLCM from GIAOVIEN --DS GVQLCM
)
--Ví dụ 2: truy vấn lồng trong from
select * from 
GIAOVIEN, (select * from DETAI) as DT
--Ví dụ 3: Xuất ra ds GV tham gia nhiều hơn 1 đề tài
select * from GIAOVIEN as GV --Lấy ra thông tin tất cả GV
where 1< -- Khi số lượng đề tài GV tham gia > 1
(
  select COUNT(*) from THAMGIADT
  where MAGV=GV.MAGV
)
--Ví dụ 3: Xuất ra thông tin khoa có nhiều hơn 2 GV
--B1:Lấy được danh sách BM nằm trong khoa hiện tại
select * from KHOA as K
where 2<
(
   select COUNT (*)from BOMON as BM, GIAOVIEN as GV
   where BM.MAKHOA=K.MAKHOA
   and BM.MABM=GV.MABM
)
--Check:
select * from KHOA as K, BOMON as BM, GIAOVIEN as GV
where BM.MAKHOA=K.MAKHOA
and BM.MABM=GV.MABM
--Sắp xếp tăng dần/giảm dần: ascending/descending:
select MAGV from GIAOVIEN
order by MAGV asc
--