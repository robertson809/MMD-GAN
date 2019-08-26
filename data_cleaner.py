import json
import array as arr
import numpy as np
filename = "/Users/Michael/Desktop/SummerPhyiscs/run100.json"
electron_momenta = []

def main():
    with open(filename) as jfile:
        data = json.load(jfile)
        for i in range(len(data['events'])):
            momentum = data['events'][str(i)]['particles'][0][1:]
            electron_momenta.append(momentum)

        arr = np.asarray(electron_momenta)
        np.save('electon.npy', arr)



def pretty_print():
    for i in range(len(data['events'])):
        print ('Electron momenta for event ', i)
        for num in data['events'][str(i)]['particles'][0][1:]:
            print (num)
        print ('')
main()

# import array as arr
# numbers = arr.array('i', [1, 2, 3])
# numbers.append(4)
# print(numbers)     # Output: array('i', [1, 2, 3, 4])
# # extend() appends iterable to the end of the array
# numbers.extend([5, 6, 7])
