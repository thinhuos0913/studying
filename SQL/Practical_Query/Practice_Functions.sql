-- 1. Với 1 mã sinh viên và 1 mã khoa, kiểm tra xem sinh viên có thuộc khoa này không (trả về đúng hoặc sai)
create function KF_Check_SV_In_Khoa
(
	@masv varchar(10),
	@makhoa varchar(10)
)
returns varchar(5)
as
begin
declare @ketqua varchar(5);

set @ketqua = 'False'
if (exists (select * from Sinh_Vien
			LEFT JOIN Lop ON Lop.Ma_Lop = Sinh_Vien.Ma_Lop
			LEFT JOIN Khoa ON Lop.Ma_Khoa = Khoa.Ma_Khoa
			where Sinh_Vien.MaSV = @masv
			and Khoa.Ma_Khoa = @makhoa
			)
	)
	set @ketqua = 'True'
else
	set @ketqua = 'False'
return @ketqua
--print @ketqua
end
go
select dbo.KF_Check_SV_In_Khoa('0212003', 'CNTT')
select dbo.KF_Check_SV_In_Khoa('0212002', 'VL')
select dbo.KF_Check_SV_In_Khoa('0212003', 'CNTT')
go
-- 2.Tính điểm thi sau cùng của một sinh viên trong một môn học cụ thể
create function KF_Last_Score_Of_Student
(
	@masv varchar(10),
	@mamh varchar(10)
)
returns float
as
begin
	declare @diem float;
	set @diem = 0;

	select top 1 @diem = Ket_Qua.Diem_Thi from Ket_Qua
	where MaSV = @masv 
	and MaMH = @mamh 
	order by Ket_Qua.Lan_Thi desc
	
	return @diem;
end
go

select dbo.KF_Last_Score_Of_Student('0212002', 'THCS01')
select dbo.KF_Last_Score_Of_Student('0212002', 'THT01')
select dbo.KF_Last_Score_Of_Student('0212002', 'THT02')
go

-- 3. Tính điểm trung bình của một sinh viên (chú ý : điểm trung bình được tính dựa trên lần thi sau cùng), sử dụng function 2 đã viết
create function KF_Last_AVG_Score_Of_Student
(
	@masv varchar(10)
)
returns float
as
begin
	declare @mamh varchar(10)
	declare @ketqua float
	set @masv = '0212002'

	select @ketqua = avg(dbo.KF_Last_Score_Of_Student(@masv, Ket_Qua.MaMH)) from Ket_Qua
	where Ket_Qua.MaSV = @masv

	return @ketqua
end
go

select dbo.KF_Last_AVG_Score_Of_Student('0212002')
go

-- 4. Nhập vào 1 sinh viên và 1 môn học, trả về các điểm thi của sinh viên này trong các lần thi của môn học đó.
create function KF_Last_List_Score_Of_Student
(
	@masv varchar(10),
	@mamh varchar(10)
)
returns table
return select Ket_Qua.Lan_Thi, Ket_Qua.Diem_Thi from Ket_Qua
	where Ket_Qua.MaSV = @masv
	and Ket_Qua.MaMH = @mamh
go

select * from dbo.KF_Last_List_Score_Of_Student('0212002', 'THCS01')