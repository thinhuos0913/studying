use Quan_Ly_Sinh_Vien
go
-- 1. Mã CT chỉ có thể là CQ,CD,TC
-- Dùng CHECK:
alter table ChuongTrinhHoc
add constraint check_Ma_CT check (MaCT in ('CQ','CD','TC')) 
go
-- Dùng TRIGGER cho case insert 1 row:
create trigger Check_MaCT
on ChuongTrinhHoc
for Update, Insert
as
begin
	declare @mact varchar(10)
	select @mact = MaCT from inserted
	if (@mact not in ('CQ','CD','TC'))
		rollback
end
--2. Học kỳ chỉ có thể là 'HK1' or 'HK2':
alter table GiangKhoa
alter column HocKy varchar(3)
go
update GiangKhoa
set HocKy = 'HK1'
where HocKy = '1'
go
update GiangKhoa
set HocKy = 'HK2'
where HocKy = '2'
go
alter table GiangKhoa
add constraint check_HK check (HocKy in ('HK1','HK2'))
go
---
create trigger Check_HocKy
on GiangKhoa
for Update, Insert
as
begin
	declare @hk varchar(3)
	select @hk = HocKy from inserted
	if (@hk not in ('HK1','HK2'))
		rollback
end
go
-- 3. Số tiết lý thuyết tối đa 120
alter table GiangKhoa
add constraint check_STLT check (STLT <= 120)
go
-- Câu 4,5 tương tự câu 3
-- 6. Điểm trong Kết quả chấm theo thang điểm 10
-- và chính xác đến 0.5 (làm bằng 2 cách, kiểm tra
-- và báo lỗi nếu không đúng quy định, tự động làm tròn
-- nếu không đúng quy định về độ chính xác)
ALTER TABLE dbo.KetQua
ADD CONSTRAINT check_diem
CHECK (Diem >= 0 AND Diem <= 10)
GO
-- Quy ước làm tròn đến 0.5
-- Dưới 0.25 làm tròn xuống .0
-- Từ 0.25 đến dưới 0.75 làm tròn thành 0.5
-- Từ 0.75 trở lên làm tròn lên 1.0
CREATE FUNCTION UF_LamTronDiem05 ( @diem FLOAT )
RETURNS FLOAT
AS
    BEGIN
        DECLARE @int INT
        SET @int = FLOOR(@diem)
        IF ( @diem - @int < 0.25 )
            SET @diem = @int
        ELSE
            IF ( @diem - @int >= 0.25
                 AND @diem - @int < 0.75
               )
                SET @diem = @int + 0.5
            ELSE
                SET @diem = @int + 1
        RETURN @diem
    END
GO

CREATE TRIGGER UTG_Diem ON dbo.KetQua
    FOR INSERT, UPDATE
AS
    BEGIN
        UPDATE  dbo.KetQua
        SET     Diem = dbo.UF_LamTronDiem05(I.Diem)
        FROM    dbo.KetQua KQ
                JOIN Inserted I ON I.Diem = KQ.Diem
                                   AND I.LanThi = KQ.LanThi
                                   AND I.MaMH = KQ.MaMH
                                   AND I.MaSV = KQ.MaSV	
        /* IF ( EXISTS ( SELECT    *
                      FROM      Inserted
                      WHERE     Diem < 0
                                OR Diem > 10 ) )
            BEGIN
                RAISERROR(N'Điểm không hợp lệ', 16, 1)
                ROLLBACK TRAN
            END */
    END
GO

-- 7. Năm kết thúc phải lớn hơn hoặc bằng Năm bắt đầu trong bảng Khóa học
ALTER TABLE dbo.KhoaHoc
ADD CONSTRAINT check_namkt
CHECK (NamKT >= NamBD)
GO

CREATE TRIGGER UTG_NamKT ON dbo.KhoaHoc
    FOR INSERT, UPDATE
AS
    BEGIN
        IF ( EXISTS ( SELECT    *
                      FROM      Inserted
                      WHERE     NamKT < NamBD ) )
            BEGIN
                RAISERROR(N'Năm kết thúc không hợp lệ', 16, 1)
                ROLLBACK TRAN
            END
    END
GO

-- 8. Số tiết lý thuyết của mỗi Giảng khoa không nhỏ hơn Số tiết thực hành
ALTER TABLE dbo.GiangKhoa
ADD CONSTRAINT check_stlt_2
CHECK (STLT >= STTH)
GO

CREATE TRIGGER UTG_STLT_2 ON dbo.GiangKhoa
    FOR INSERT, UPDATE
AS
    BEGIN
        IF ( EXISTS ( SELECT    *
                      FROM      Inserted
                      WHERE     STLT < STTH ) )
            BEGIN
                RAISERROR(N'Số tiết lý thuyết không hợp lệ', 16, 1)
                ROLLBACK TRAN
            END
    END
GO
