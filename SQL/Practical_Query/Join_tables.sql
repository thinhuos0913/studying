use sqltutorial
go

select * 
from EmployeeSalary

select * 
from EmployeeDemographics

select FirstName,LastName,Salary
from EmployeeDemographics
join EmployeeSalary
on EmployeeDemographics.EmployeeID = EmployeeSalary.EmployeeID
where Salary > 45000

select count(*)
from EmployeeDemographics
join EmployeeSalary
on EmployeeDemographics.EmployeeID = EmployeeSalary.EmployeeID
--where Salary > 45000