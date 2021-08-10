import sys

index = int(sys.argv[1])
inputFile = str(sys.argv[2])

def lineToNums(line):
    words = line.split(' ')
    nums = map(float,words)
    return nums


colNums=list()
with open(inputFile,'r') as f:
    for line in f.readlines():
        line = line.strip()
        if line != "":
            nums = list(lineToNums(line))
            colNums.append(nums[index])


sortedColNums = sorted(colNums)
output=','.join(map(str,sortedColNums))
print(output)
open(r"./result/ans1.dat",'w').write(output)