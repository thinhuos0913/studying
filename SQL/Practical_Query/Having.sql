use HowKteam
go
select BOMON.MABM, COUNT(*)as 'SL' from GIAOVIEN, BOMON
where BOMON.MABM=GIAOVIEN.MABM
group by BOMON.MABM
having COUNT(*)>1
--Vd2: Xuat ra muc luong va tong tuoi cua GV nhan muc luong do 
select LUONG, SUM(Year(getdate()) - Year(NGSINH)) from GIAOVIEN
group by LUONG
--Code:
USE HowKteam
GO

-- xuất ra số lượng giáo viên trong từng bộ môn mà số giáo viên > 2
-- having -> như where của select nhưng giành cho group by
-- having là where của group by
SELECT dbo.BOMON.MABM, COUNT(*) FROM dbo.GIAOVIEN, dbo.BOMON
WHERE dbo.BOMON.MABM = GIAOVIEN.MABM 
GROUP BY dbo.BOMON.MABM
HAVING COUNT(*) > 1


-- Xuất ra mức lương và tổng tuổi của giáo viên nhận mức lương đó
-- và có người thân
-- và tuổi phải lớn hơn tuổi trung bình
SELECT LUONG, SUM(YEAR(GETDATE()) - YEAR(GIAOVIEN.NGSINH)) FROM dbo.GIAOVIEN, dbo.NGUOITHAN
WHERE GIAOVIEN.MAGV = NGUOITHAN.MAGV
AND GIAOVIEN.MAGV IN (SELECT MaGV FROM dbo.NGUOITHAN)
GROUP BY LUONG, GIAOVIEN.NGSINH
HAVING YEAR(GETDATE()) - YEAR(GIAOVIEN.NGSINH) > 
(
	(SELECT SUM(YEAR(GETDATE()) - YEAR(GIAOVIEN.NGSINH)) FROM dbo.GIAOVIEN)
	/ (SELECT COUNT(*) FROM dbo.GIAOVIEN)
)