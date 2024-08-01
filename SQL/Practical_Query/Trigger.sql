USE HowKteam 
GO
-- CREATE FUNCTION PRINT_NAME (@Name nvarchar(100))
CREATE TRIGGER INSERT_GV
ON GIAOVIEN
FOR INSERT 
AS
BEGIN
	PRINT 'TRIGGER'
END
go
--Example:
alter trigger AbortOlderThan40 -- alter: to edit Trigger
on GiaoVien
for delete
as
begin
	declare @count int = 0
	select @count = count(*) from deleted
	where year(getdate()) - year(deleted.NgSinh) > 40
	if (@count > 0)
	begin
		print 'Access Denied!'
		rollback tran -- ko cho phep xoa GV > 40 tuoi
	end
end
go
select * from GiaoVien
insert GiaoVien (MAGV,HOTEN,LUONG,PHAI,NGSINH,DIACHI,GVQLCM,MABM) --insert 1 GV > 40 tuoi de check
		values ('011','Daniels',2000,'Nam','19790724','San Jose',null,'MMT')
select * from GiaoVien
delete GiaoVien where MaGV = '011'
---
select distinct GiaoVien.* from GiaoVien
--select * from GV_DT
left join GV_DT on GIAOVIEN.MAGV = GV_DT.MAGV

-- Test trường hợp insert dữ liệu đồng loạt vào bảng (Vi du thuc te tham khao)
-- TẠO MỘT TRIGGER MỚI TEST TRƯỜNG HỢP THÊM DỮ LIỆU ĐỒNG LOẠT

CREATE TRIGGER UTG_Insert1 ON TEST FOR INSERT, UPDATE
AS
BEGIN
-- KHỐI LỆNH KIỂM TRA THEO Ý MUỐN, Ở ĐÂY TA THÔNG BÁO KHI CÓ MỘT TRƯỜNG MỚI ĐƯỢC THÊM VÀO

    DECLARE @TK INT
    SELECT @TK = inserted.TAIKHOAN FROM inserted
    IF @TK < 1000
    BEGIN
        PRINT N'KHÔNG ĐỦ TIỀN'
        ROLLBACK TRAN
    END
    ELSE
    BEGIN
        PRINT N'TÀI KHOẢN ĐÃ ĐƯỢC THÊM'
    END
END

CREATE TABLE test
(
    STT INT IDENTITY, -- TỰ TĂNG THỨ TỰ
    HOTEN NVARCHAR(50),
    TAIKHOAN FLOAT
)


DECLARE @I INT = 0
DECLARE @J INT = 10
DECLARE @balance INT = 1001
WHILE(@I<@J)
BEGIN
    SELECT @balance = TAIKHOAN FROM dbo.test
    INSERT INTO dbo.test
    (
        HOTEN,
        TAIKHOAN
    )
    VALUES
    (   N'test', -- HOTEN - nvarchar(50)
        @balance + 3  -- TAIKHOAN - float
        )
    SET @i += 1
END