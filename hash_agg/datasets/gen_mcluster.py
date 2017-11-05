import random
import math
import sys

num_groups = int(math.pow(2, int(sys.argv[1])))
num_items = 64
for j in range(512*1024):
    for i in range(num_items):
        print random.randint(j*num_groups/1024/1024, j*num_groups/1024/1024+16), random.randint(1, 5)




