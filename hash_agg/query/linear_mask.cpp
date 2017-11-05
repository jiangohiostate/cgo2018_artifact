#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "immintrin.h"
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

int num_bits[65536];
__m512i gather_indices[65536];

int *test_base;

__m512i v_my_hash(__m512i vkey)
{
  __m512i t = _mm512_mullo_epi32(vkey, _mm512_set1_epi32(T));
  return _mm512_srlv_epi32(t, _mm512_set1_epi32(SHIFT));
}

int my_hash(int key)
{
  const long long w_bit = 4294967295;
  long long r0 = key * T;
  return ((r0 & w_bit)) >> SHIFT;
}

void print_vec(__m512i v)
{
  for(int i=0;i<16;i++) {
    cout << *((int *)&v + i) << " ";
  }
  cout << endl;
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
  test_base = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_count = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_sum = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);
  h_squaresum = (int *)_mm_malloc(sizeof(int)*H_SIZE, 64);

  for(int i=0;i<H_SIZE;i++) {
    h_keys[i] = -1;
    test_base[i] = -1;
  }

  for(int i=0;i<65536;i++) {
    int c = 0;
    int t = i;
    int j = 0;
  
    __declspec(align(64)) int temp[16];
    for(int k=0;k<16;k++) temp[k] = 0;
    while (t)
    {
      if(t & 1) {
        temp[j] = c++;
      };
      j++;
      t /= 2;
    }
    num_bits[i] = c;
    gather_indices[i] = _mm512_load_epi32(temp);
  }


  cout << "input finish! " << endl;

  __m512i voffset = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  __m512i vnone = _mm512_set1_epi32(-1);
  __m512i vone = _mm512_set1_epi32(1);
  __m512i vzero = _mm512_set1_epi32(0);
  __m512i vupper = _mm512_set1_epi32(H_SIZE);

  __m512i vbound = _mm512_set1_epi32(cur);


  long step = 0;
  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  gettimeofday(&tv1, &tz1);

  __mmask16 m = 65535;
  __m512i vkey;
  __m512i vvalue;
  int i = 0;

  while(i < cur) {

    __m512i vindex = gather_indices[m];
    vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(i));

    m = _mm512_mask_cmplt_epi32_mask(m, vindex, vbound);

    vkey = _mm512_mask_i32gather_epi32(vkey, m, vindex, keys, 4);
    vvalue = _mm512_mask_i32gather_epi32(vvalue, m, vindex, values, 4);
    i += num_bits[m];

    step++;


    __m512i vh = v_my_hash(vkey);


 //   _mm512_prefetch_i32scatter_ps(h_sum, vh, 4, _MM_HINT_T0);
  //  _mm512_prefetch_i32scatter_ps(h_count, vh, 4, _MM_HINT_T0);
   // _mm512_prefetch_i32scatter_ps(h_squaresum, vh, 4, _MM_HINT_T0);

    __m512i vhkey = _mm512_i32gather_epi32(vh, h_keys, 4);

    __mmask16 m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
    __mmask16 m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);

    while(m1 != m2) {
      vh = _mm512_mask_add_epi32(vh, ~m2&m1, vh, vone);
      __mmask mx = _mm512_mask_cmpgt_epi32_mask(~m2&m1, vh, vupper);
      vh = _mm512_mask_mov_epi32(vh, mx, vzero); 
      vhkey = _mm512_mask_i32gather_epi32(vhkey, ~m2&m1, vh, h_keys, 4);
      m1 = _mm512_mask_cmpneq_epi32_mask(m, vhkey, vnone);
      m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);
    }

    _mm512_i32scatter_epi32(h_keys, vh, voffset, 4);
    __m512i vtest = _mm512_i32gather_epi32(vh, h_keys, 4);
    __mmask16 mnoconf = _mm512_cmpeq_epi32_mask(vtest, voffset);
    _mm512_i32scatter_epi32(h_keys, vh, vhkey, 4);
    m = mnoconf;

    __m512i vsquare = _mm512_mullo_epi32(vvalue, vvalue);
    __m512i vsum = vvalue;
    __m512i vcount = vone;
    __m512i vsquaresum = vsquare;

    if(m1!=65535) {

      _mm512_mask_i32scatter_epi32(h_keys, ~m1&m, vh, vkey, 4);
      _mm512_mask_i32scatter_epi32(h_sum, ~m1&m, vh, vvalue, 4);
      _mm512_mask_i32scatter_epi32(h_count, ~m1&m, vh, vone, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, ~m1&m, vh, vsquare, 4);
    }

    if(m1!=0) {
      vsum = _mm512_mask_i32gather_epi32(vsum, m1&m, vh, h_sum, 4);
      vcount = _mm512_mask_i32gather_epi32(vcount, m1&m, vh, h_count, 4);
      vsquaresum = _mm512_mask_i32gather_epi32(vsquaresum, m1&m, vh, h_squaresum, 4);

      vsum = _mm512_mask_add_epi32(vsum, m1&m, vsum, vvalue);
      vcount = _mm512_mask_add_epi32(vcount, m1&m, vcount, vone);
      vsquaresum = _mm512_mask_add_epi32(vsquaresum, m1&m, vsquaresum, vsquare);

      _mm512_mask_i32scatter_epi32(h_sum, m1&m, vh, vsum, 4);
      _mm512_mask_i32scatter_epi32(h_count, m1&m, vh, vcount, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, m1&m, vh, vsquaresum, 4);
    }



  }

  for(int i=0;i<16;i++) {
    if(~m & (1<<i)) {
      int key = *((int *)(&vkey)+i);
      int value = *((int *)(&vvalue)+i);
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

  }




  gettimeofday(&tv2, &tz2);
  cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  cout << "simd utilization: " << 1.0 * cur / step / 16 << endl;

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
