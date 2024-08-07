use SQLTutorial
go

select *
from EmployeeSalary

-- Find the employee with 2nd highest salary
with ranking as 
(
	select EmployeeID,
			Salary,
			DENSE_RANK() over (order by Salary desc) as SalaryRank
	from EmployeeSalary
)

select EmployeeID,
		Salary,
		SalaryRank
from ranking
where SalaryRank = 2