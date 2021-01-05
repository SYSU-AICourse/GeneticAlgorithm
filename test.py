a = list(range(10))
a[1:9] = a[8:0:-1]
print(a)