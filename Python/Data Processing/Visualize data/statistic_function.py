import math

data = [0.0, 0.25, 0.25, 1.25, 1.5, 1.75, 2.75, 3.25, 100]
# MEAN
def mean(data):
    return sum(data)/len(data)

mu = mean(data) # mu - đọc là muy, là tên tiếng Anh của chữ latin: μ
print(mu)

# MEDIAN
def median(data):
    data = sorted(data)
    n = len(data)
    if n % 2 == 1:
        return data[n//2]
    else:
        i = n//2
        return (data[i - 1] + data[i])/2

med = median(data)
print(med)

# MEDIAN LOW and HIGH
def median_low(data):
    data = sorted(data)
    n = len(data)
    if n % 2 == 1:
        return data[n//2]
    else:
        return data[n//2-1]

def median_high(data):
    data = sorted(data)
    n = len(data)
    return data[n//2]

med_low = median_low(data)
print(med_low)
med_high = median_high(data)
print(med_high)

# MODE: Giá trị xuất hiện nhiều lần nhất trong tập dữ liệu
def mode(data):
    dmax = data[0]
    for d in data:
        if data.count(d) > data.count(dmax):
            dmax = d
    return dmax

print(mode(data))
print(mode(["red", "blue", "blue", "red", "green", "red", "red"]))

# DEVIATION
# def mydev(data):
#     mu = mean(data)
#     return sum([point-mu for point in data])/len(data)

# print(mydev([1,5,9]))

def mydev(data): # this is MAD 
    mu = mean(data)
    return sum([abs(point-mu) for point in data])/len(data)

print(mydev([1,5,9]))
# Với giá trị 2.6666666666666665, có thể hiểu rằng các điểm trong tập dữ liệu được phân bố cách điểm trung tâm (ở ví dụ này là mean) một khoảng trung bình là ~2.6
print(mydev(data))

# VARIANCE and STANDARD DEVIATION
def variance(data):
    mu = mean(data)
    return sum([(point-mu)**2 for point in data])/len(data)

print(variance([1,5,9]))
print(variance(data))

def stddev(data): # standard deviation
    return math.sqrt(variance(data))

print(stddev([1,5,9]))
print(stddev(data))