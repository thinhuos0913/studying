use HowKteam
go
select * from GIAOVIEN where LUONG = (Select(MAX(LUONG)) from GIAOVIEN)
Declare @MaxSalary char(10)
Select @MaxSalary = MAGV from GIAOVIEN where LUONG = (Select(MAX(LUONG)) from GIAOVIEN)
Select YEAR(Getdate()) - YEAR(NGSINH) from GIAOVIEN where MAGV = @MaxSalary
Declare @MaGV char(10) = '001'
Select COUNT(*) from NGUOITHAN where MAGV = @MaGV
PRINT @MaGV
