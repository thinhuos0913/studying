Use HowKteam
Go
Declare @LuongTB int 
Declare @SoluongGV int
Select @SoluongGV = COUNT(*) from GIAOVIEN
Select @LuongTB = SUM(LUONG)/@SoluongGV from GIAOVIEN
Declare @MaGV char(10) = '006'
Declare @LuongMaGV int = 0
Select @LuongMaGV = LUONG from GIAOVIEN where MAGV = @MaGV
IF @LuongMaGV > @LuongTB
	PRINT N'Greater than Average salary'
ELSE
	PRINT N'Less than Average salary'
Select * from GIAOVIEN
-- Update luong GV neu luong > luongTB, else update luong cua GV nu
Declare @LuongTB int 
Declare @SoluongGV int
Select @SoluongGV = COUNT(*) from GIAOVIEN
Select @LuongTB = SUM(LUONG)/@SoluongGV from GIAOVIEN
Declare @Luong int = 1500
If (@Luong > @LuongTB)
Begin
	Update GIAOVIEN Set LUONG = @Luong
End
Else
Begin
	Update GIAOVIEN Set LUONG = @Luong where PHAI = N'Nữ'
End
Select * from GIAOVIEN