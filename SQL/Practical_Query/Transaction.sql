use HowKteam
go
select * from NGUOITHAN where TEN = 'Thu'
select * from NGUOITHAN
begin transaction
delete NGUOITHAN where TEN=N'Thủy'
rollback -- Denied Transaction
-- ACCEPT TRANSACTION
begin transaction
delete NGUOITHAN where TEN=N'Thủy'
commit -- Accept Transaction
-- Naming TRANSACTION
declare @Trans varchar(20)
select @Trans = 'Trans1'
begin transaction @Trans
delete NGUOITHAN where TEN = N'Thy'
commit transaction
-----
begin transaction
save transaction Trans1 -- Save Point to come back

delete NGUOITHAN where TEN = 'George'

save transaction Trans2

delete NGUOITHAN where TEN = 'Thu'
rollback Transaction Trans2
-- insert data to test
insert NGUOITHAN (MAGV,TEN,NGSINH,PHAI)
values ('001','George','19900725','Nam')
go