#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <stdlib.h>
using namespace std;

int *keys;
int *values;


#define T 2654435769 

#if defined SMALL
#define SHIFT 20
#define H_SIZE 4096
#elif defined MEDIUM
#define SHIFT 16
#define H_SIZE 65536
#elif defined LARGE
#define SHIFT 12
#define H_SIZE 1048576
#elif defined XLARGE
#define SHIFT 8
#define H_SIZE 16777216

#endif

int *h_keys;
int *h_sum;
int *h_count;
int *h_squaresum;

int my_hash(int key)
{
  const long long w_bit = 4294967295;
  long long r0 = key * T;
  return ((r0 & w_bit)) >> SHIFT;
}

int main(int argc, char *argv[])
{

  keys = (int *)_mm_malloc(sizeof(int)*32*1024*1024, 64);
  values = (int *)_mm_malloc(sizeof(int)*32*1024*1024, 64);

  ifstream fin(argv[1]);
  int a, b;
  int cur = 0;
  while(fin >> a >> b) {
    keys[cur] = a;
    values[cur] = b;
    cur++;
  }

  h_keys = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_count = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_sum = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_squaresum = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);

  for(int i=0;i<H_SIZE;i++) {
    h_keys[i] = -1;
  }


  cout << "input finish! " << endl;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  gettimeofday(&tv1, &tz1);

  for(int i=0;i<cur;i++) {
    int key = keys[i];
    int value = values[i];
    int h = my_hash(key);
    bool flag = false;
    while(h_keys[h]!=-1) {
      if(key == h_keys[h]) {
          h_sum[h] += value;  
          h_count[h]++;
          h_squaresum[h] += value * value;
        flag = true;
        break;
      }
      h++;
      if(h==H_SIZE)h = 0;
    }
    if(!flag) {
      h_keys[h] = key;
      h_sum[h] = value;
      h_count[h] = 1;
      h_squaresum[h] = value * value;
    }

  }
  gettimeofday(&tv2, &tz2);
  cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;

  int h = my_hash(keys[0]);
  while(h_keys[h]!=-1) {
    if(keys[0] == h_keys[h]) {
      cout << h_count[h] << " " << h_sum[h] << " " << h_squaresum[h] << endl;
      break;
    }
    h++;
    if(h==H_SIZE)h = 0;
  }

  return 0;

}
