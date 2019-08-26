import numpy as np
import sys
#print(sys.argv[0])
data_path = sys.argv[1]
print(data_path)
data = np.load(data_path).astype(np.float32)
count = 0
print("shape is", data.shape)
for row in data:
    #row[2] = 0.0
    if count > 100:
        break
    print(count)
    count += 1

    print(row)
    #print ("")
    #z = np.array([0,0,1])
    #row = row - np.dot(z,row)*np.array([0,0,1]))


# with open(filename) as datafile:
#    data = np.loadtxt(datafile)
#     for row in file:
#         print(type(row))

#55.947625
