create database SQLBQUERY
use SQLBQUERY
create table QuanLyNhanVien
(
 MaNV int,
 HoVaTen varchar(100),
 NgaySinh Date,
 Tuoi int,
 ViTri varchar(100),
 GioiTinh bit,
)
alter table QuanLyNhanVien --Sửa bảng QuanLyNhanVien
drop column GioiTinh --Xóa cột GioiTinh
alter table QuanLyNhanVien add Nam bit --Them cot gioi tinh Nam:1-true,0-false
insert into QuanLyNhanVien
values 
(
 '3',--Ma NV
 'Luca Pellegrini',--HoVaTen
 '19970708',--NgaySinh
 '21',--Tuoi
 'LB',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '3',--Ma NV
 'Luca Pellegrini',--HoVaTen
 '19970708',--NgaySinh
 '21',--Tuoi
 'LB',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '6',--Ma NV
 'Kevin Strootman',--HoVaTen
 '19900213',--NgaySinh
 '29',--Tuoi
 'CM',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '4',--Ma NV
 'Bryan Cristante',--HoVaTen
 '19950612',--NgaySinh
 '22',--Tuoi
 'CM',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '5',--Ma NV
 'Juan Jesus',--HoVaTen
 '19910808',--NgaySinh
 '27',--Tuoi
 'LCB',--ViTri
 '1'--GioiTinh
)
delete QuanLyNhanVien where MaNV=6 --Xoa nhan vien co MaNV=6
update QuanLyNhanVien set ViTri='CB' where MaNV=5 --Update vi tri nhan vien co MaNV=5
insert into QuanLyNhanVien
values 
(
 '7',--Ma NV
 'Lo. Pellegrini',--HoVaTen
 '19960902',--NgaySinh
 '22',--Tuoi
 'AM',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '8',--Ma NV
 'Diego Perotti',--HoVaTen
 '19880712',--NgaySinh
 '30',--Tuoi
 'LW',--ViTri
 '1'--GioiTinh
)
insert into QuanLyNhanVien
values 
(
 '9',--Ma NV
 'Edin Dzeko',--HoVaTen
 '19860619',--NgaySinh
 '32',--Tuoi
 'CF',--ViTri
 '1'--GioiTinh
)
alter table QuanLyNhanVien
drop column MaNV
alter table QuanLyNhanVien add ID int not null unique
create table Test1
(
ID int not null unique,
Pos varchar(10),
Club varchar(10),
)
alter table Test1 add primary key (ID) --Tao khoa chinh cho cot ID
alter table QuanLyNhanVien add primary key (HoVaTen)