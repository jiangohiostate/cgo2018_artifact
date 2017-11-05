import random 
import bisect 
import math 
import sys

class ZipfGenerator: 
  def __init__(self, n, alpha): 
    tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)] 
    zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0]) 
    self.distMap = [x / zeta[-1] for x in zeta] 

  def next(self): 
      u = random.random()  
      return bisect.bisect(self.distMap, u) - 1


num_groups = int(math.pow(2, int(sys.argv[1])))
num_items = 32*1024*1024;
zg = ZipfGenerator(num_groups, 2)
for i in range(num_items):
    print zg.next(), random.randint(1, 3)
