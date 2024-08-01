Use HowKteam 
Go
Declare @i int = 100
Declare @n int = 10100
While (@i < @n)
Begin
	Insert BOMON
		(
			MABM,
			TENBM,
			PHONG,
			DIENTHOAI,
			TRUONGBM,
			MAKHOA,
			NGAYNHANCHUC
		)
	Values (@i,--
			@i,--
			'B15',--
			'0988607818',--
			null,--
			N'CNTT',--
			GETDATE()--
			)
	Set @i += 1
End
Select * from BOMON