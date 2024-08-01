Use HowKteam
Go
CREATE PROC USP_Test
@MaGV nvarchar(10), @Luong int -- Pass argument @MaGV and @Luong
AS
BEGIN
	SELECT * FROM GIAOVIEN WHERE MAGV = @MaGV AND LUONG = @Luong
END
GO
EXEC USP_Test @MaGV = N'',@Luong = 0
EXEC USP_Test N'',0
EXECUTE USP_Test @MaGV = N'',@Luong = 0
EXECUTE USP_Test N'',0
GO
----
CREATE PROC USP_GV
--Not pass arguments
AS SELECT * FROM GIAOVIEN
GO
EXEC USP_GV 

USE SQLTutorial
GO

CREATE PROC Salary_Test
@EmpID nvarchar(10), @Salary int
AS 
BEGIN
	SELECT *
	FROM EmployeeSalary
	WHERE EmployeeID = @EmpID AND Salary = @Salary
END

EXEC Salary_Test 
1001,45000

select *
from EmployeeSalary
