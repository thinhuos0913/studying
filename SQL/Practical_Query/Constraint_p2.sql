Use Quan_Ly_Sinh_Vien
Go
-- 1. Tên CT phải phân biệt
alter table ChuongTrinhHoc
add constraint unique_TenCT unique(TenCT)
--2. Tên Khoa phải duy nhất
alter table Khoa
add constraint unique_TenKhoa unique(TenKhoa)
-- 3. Tương tự câu 1,2
-- 4. SV chỉ được thi tối đa 2 lần 1 môn:
alter table KetQua 
add constraint check_LanThi CHECK(LanThi <= 2)