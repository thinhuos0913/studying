create database Quan_Ly_Sinh_Vien
go
use Quan_Ly_Sinh_Vien
go
create table Khoa
(
	MaKhoa varchar(10) primary key,
	TenKhoa nvarchar(100),
	NamThanhLap int
)
go
create table KhoaHoc
(
	MaKhoaHoc varchar(10) primary key,
	NamBatDau int,
	NamKetThuc int
)
go
create table ChuongTrinhHoc
(
	MaCT varchar(10) primary key,
	TenCT nvarchar(100),
)
go
create table Lop
(
	MaLop varchar(10) primary key,
	MaKhoa varchar(10) not null,
	MaKhoaHoc varchar(10) not null,
	MaCT varchar(10) not null,
	STT int
	foreign key(MaKhoa) references Khoa(MaKhoa),
	foreign key(MaKhoaHoc) references KhoaHoc(MaKhoaHoc),
	foreign key(MaCT) references ChuongTrinhHoc(MaCT)
)
go
create table SinhVien
(
	MaSV varchar(10) primary key,
	HoTen varchar(100) not null,
	NamSinh int,
	DanToc varchar(20),
	MaLop varchar(10) not null
	foreign key(MaLop) references Lop(MaLop),
)
go
create table MonHoc
(
	MaMH varchar(10) primary key,
	MaKhoa varchar(10) not null,
	TenMH nvarchar(100)
	foreign key(MaKhoa) references Khoa(MaKhoa),
)
go
create table KetQua
(
	MaSV varchar(10) not null,
	MaMH varchar(10) not null,
	LanThi int not null,
	DiemThi float
	
	primary key(MaSV,MaMH,LanThi)
	
	foreign key(MaSV) references SinhVien(MaSV),
	foreign key(MaMH) references MonHoc(MaMH),
)
go
create table GiangKhoa
(
	MaCT varchar(10) not null,
	MaKhoa varchar(10) not null,
	MaMH varchar(10) not null,
	NamHoc int not null,
	HocKy int,
	STLT int,
	STTH int,
	SoTinChi int
	
	primary key(MaCT,MaKhoa,MaMH,NamHoc)
	
	foreign key(MaCT) references ChuongTrinhHoc(MaCT),
	foreign key(MaKhoa) references Khoa(MaKhoa),
	foreign key(MaMH) references MonHoc(MaMH)
)
go
