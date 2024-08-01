Use HowKteam
Go
-- Create a function select all gv
CREATE FUNCTION SELECT_ALL_GV()
RETURNS TABLE
AS RETURN SELECT * FROM GIAOVIEN
GO
SELECT * FROM SELECT_ALL_GV()

-- Function with parameters
CREATE FUNCTION SELECT_LUONG_GV(@MaGV char(10))
RETURNS INT
AS
BEGIN 
	DECLARE @Luong int
	SELECT @Luong = Luong FROM GIAOVIEN WHERE MAGV = @MaGV
	RETURN @Luong
END

--Use function
SELECT dbo.SELECT_LUONG_GV('001') AS DBS
SELECT dbo.SELECT_LUONG_GV(MaGV) from GIAOVIEN

-- Edit function: replace CREATE by ALTER
ALTER FUNCTION SELECT_LUONG_GV(@MaGV char(10))
RETURNS INT
AS
BEGIN 
	DECLARE @Luong int
	SELECT @Luong = Luong FROM GIAOVIEN WHERE MAGV = @MaGV
	RETURN @Luong
END

-- Delete function & store proc
DROP PROC USP_GV
DROP FUNCTION SELECT_ALL_GV

-- Example:
create function Is_Odd(@Num int)
returns nvarchar(20)
as 
begin
	if(@Num % 2 = 0)
		return N'ODD'
	else 
		return N'EVEN'
	return N'UNDEFINED'
end
create function Age(@Year date)
returns int
begin
	return year(getdate()) - year(@Year)
end
select dbo.Age(NgSinh), dbo.Is_Odd(dbo.Age(NgSinh)) from GIAOVIEN
drop function DOB