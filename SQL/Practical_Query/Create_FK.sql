create database Primary_Foreign
go
use Primary_Foreign
go
create table BoMon
(
 MaBM char(10)primary key,
 Name nvarchar (100) default N'Tên BM'
)
go
create table Lop
(
 MaLop char (10) not null,
 Name nvarchar (100) default N'Tên lớp'
)
go
use Primary_Foreign
go
alter table Lop add primary key (MaLop)
create table GiaoVien
(
 MaGV char (10) not null,
 Name nvarchar (100) default N'Tên GV',
 DiaChi nvarchar (100) default N'Địa chỉ GV',
 NgaySinh date,
 Sex bit,
 MaBM char(10)
 ---Tạo khóa ngoại ngay khi tạo bảng
 foreign key (MaBM) references BoMon(MaBM)
)
go
use Primary_Foreign
go
alter table GiaoVien add foreign key (MaBM) references BoMon(MaBM)
alter table GiaoVien add MaBM char(10)
alter table GiaoVien add primary key (MaGV)

create table HocSinh
(
 MaHS char (10) primary key,
 Name nvarchar (100),
 MaLop char (10)
)
go
---Tạo khóa ngoại sau khi tạo bảng
use Primary_Foreign
go
alter table HocSinh add MaLop char (10)
use Primary_Foreign
go
alter table HocSinh add constraint FK_HS foreign key (MaLop) references Lop (MaLop)
alter table HocSinh drop constraint FK_HS --- Hủy khóa ngoại bằng cách tạo 1 constraint FK_HS(khóa chính)
insert BoMon
values 
(
 '01',
 N'Bộ môn 1'
)
insert BoMon
values 
(
 '02',
 N'Bộ môn 2'
)
insert BoMon
values 
(
 '03',
 N'Bộ môn 3'
)
insert GiaoVien
values 
(
 'GV01',
 N'GV1',
 N'DC1',
 GETDATE(),
 1,
 '03'
)
insert GiaoVien
values 
(
 'GV02',
 N'GV2',
 N'DC2',
 GETDATE(),
 0,
 '02'
)
insert GiaoVien
values 
(
 'GV03',
 N'GV3',
 N'DC3',
 GETDATE(),
 1,
 '01'
)