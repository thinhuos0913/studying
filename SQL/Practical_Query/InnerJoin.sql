--Inner Join
use HowKteam
go
select * from GIAOVIEN, BOMON
where GIAOVIEN.MABM=BOMON.MABM --Inner Join kiểu cũ, có thể sau này sẽ không còn được hỗ trợ
select * from 
GIAOVIEN inner join BOMON
on GIAOVIEN.MABM = BOMON.MABM--Inner Join kiểu mới, Microsoft khuyến khích sử dụng
select * from 
GIAOVIEN join BOMON--có thể viết tắt Inner Join = Join
on GIAOVIEN.MABM=BOMON.MABM
--Inner join ket hop dieu kien WHERE
use HowKteam
go
select GV.MAGV, GV.HOTEN, GV.PHAI, BM.TENBM, K.TENKHOA
from BOMON as BM
join GIAOVIEN as GV on GV.MABM=BM.MABM
join KHOA as K on K.MAKHOA=BM.MAKHOA
where GV.PHAI='Nam'
