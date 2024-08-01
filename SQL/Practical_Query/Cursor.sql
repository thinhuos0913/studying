Use HowKteam
Go
--Xet tuoi GV
select * from GIAOVIEN
--Neu tuoi > 40 -> set luong = 2500, if luong < 40 & > 30 set 2000, else 1500
declare GVCursor cursor for select MAGV, year(Getdate()) - Year(NGSINH)from GIAOVIEN
open GVCursor
declare @MaGV char(10)
declare @Tuoi int
fetch next from GVCursor into @MaGV, @Tuoi
while @@FETCH_STATUS = 0
begin
	if @Tuoi > 40
	begin
		update GIAOVIEN set LUONG = 2500 where @MaGV = MAGV
	end
	else 
	if @Tuoi > 30
	begin 
		update GIAOVIEN set LUONG = 2000 where @MaGV = MAGV
	end
	else 
	begin 
		update GIAOVIEN set LUONG = 1500 where @MaGV = MAGV
	end
	fetch next from GVCursor into @MaGV, @Tuoi
end
close GVCursor
deallocate GVCursor
select * from GIAOVIEN