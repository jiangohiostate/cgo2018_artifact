#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "immintrin.h"
#include <stdlib.h>
#include <hbwmalloc.h>
using namespace std;

int *keys;
int *values;


#define T 2654435769 

//#define _mm_malloc(size, align) hbw_malloc(size)
//#define _mm_free(addr) hbw_free(addr)



#if defined SMALL
#define SHIFT 24
#define H_SIZE 256
#elif defined MEDIUM
#define SHIFT 20
#define H_SIZE 4096
#elif defined LARGE
#define SHIFT 16
#define H_SIZE 65536
#elif defined XLARGE
#define SHIFT 12
#define H_SIZE 1048576
#endif

#define B 16

int *h_keys;
int *h_sum;
int *h_count;
int *h_squaresum;

int *keys_temp[H_SIZE];
int *values_temp[H_SIZE];
int pos[H_SIZE];


int num_bits[65536];
__m512i gather_indices[65536];

void print_table() 
{
  for(int i=0;i<H_SIZE;i++) {
    for(int j=0;j<B;j++) {
      cout << h_keys[i*B+j] << " ";
    }
    cout << endl;
  }
  cout << "===================================" << endl;
}

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
  int num_records = 0;
  while(fin >> a >> b) {
    keys[num_records] = a;
    values[num_records] = b;
    num_records++;
  }

  h_keys = (int *)_mm_malloc(sizeof(int)*H_SIZE*B, 64);
  h_count = (int *)_mm_malloc(sizeof(int)*H_SIZE*B, 64);
  h_sum = (int *)_mm_malloc(sizeof(int)*H_SIZE*B, 64);
  h_squaresum = (int *)_mm_malloc(sizeof(int)*H_SIZE*B, 64);



  for(int i=0;i<H_SIZE*B;i++) {
    h_keys[i] = -1;
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


  __m512i vnone = _mm512_set1_epi32(-1);
  __m512i vone = _mm512_set1_epi32(1);
  __m512i v3 = _mm512_set1_epi32(3);
  __m512i vzero = _mm512_set1_epi32(0);
  __m512i v16 = _mm512_set1_epi32(16);
  __m512i voffset = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

  __m512i vshuffle = voffset;
  __m512i vbound = _mm512_set1_epi32(num_records);

  int overhead = 0;


  cout << "input finish! " << endl;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  gettimeofday(&tv1, &tz1);
  //struct timeval tv1, tv2;
 // struct timezone tz1, tz2;

 // gettimeofday(&tv1, &tz1);



  for(int i=0; i<num_records; i+=16) {

    __m512i vkey = _mm512_load_epi32(keys+i);
    __m512i vvalue = _mm512_load_epi32(values+i);

    __m512i vh1 = v_my_hash(vkey);
    vh1 = _mm512_mullo_epi32(vh1, _mm512_set1_epi32(B));

    __m512i vh_lower = vh1;
    __m512i vh_upper = _mm512_add_epi32(vh1, v16);

    __declspec(align(64)) int lower_temp[16];
   _mm512_store_epi32(lower_temp, vh1);

    __m512i vh = _mm512_add_epi32(vh1, vshuffle);


    __m512i vhkey = _mm512_i32gather_epi32(vh, h_keys, 4);

//    _mm512_prefetch_i32scatter_ps(h_sum, vh, 4, _MM_HINT_T0);
 //   _mm512_prefetch_i32scatter_ps(h_count, vh, 4, _MM_HINT_T0);
 //   _mm512_prefetch_i32scatter_ps(h_squaresum, vh, 4, _MM_HINT_T0);


    __mmask16 m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
    __mmask16 m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);

    int aa = 0;
    while(m1 != m2) {
      if(aa==16) {

        for(int ii=0;ii<16;ii++) {
          if(~m2&m1 & (1<<ii)) {
            int low = lower_temp[ii]; 
            for(int jj=low;jj<low+15;jj++) {
              if(h_keys[jj] == -1) {
                int kk;
                for(kk=jj+1;kk<low+16;kk++) {
                  if(h_keys[kk] != -1) {
                    h_keys[jj] = h_keys[kk];
                    h_keys[kk] = -1;
                    h_count[jj] = h_count[kk];
                    h_sum[jj] = h_sum[kk];
                    h_squaresum[jj] = h_squaresum[kk];
                    break;
                  }
                }
                if(kk==low+16) break;
              } 
              for(int kk=jj+1;kk<low+16;kk++) {
                if(h_keys[kk] == h_keys[jj]) {
                  h_keys[kk] = -1;
                  h_count[jj] += h_count[kk];
                  h_sum[jj] += h_sum[kk];
                  h_squaresum[jj] += h_squaresum[kk];
                }
              }
            }   
            if(h_keys[low+15]!=-1) {
              for(int jj=low;jj<low+16;jj++) {
                cout << h_keys[jj] << " ";
              }
              cout << endl;
              exit(1);
            }
          }
        }












      //print_table();

      aa = 0;
      vh = _mm512_mask_mov_epi32(vh, ~m2&m1, vh_lower);
      } else {
        vh = _mm512_mask_add_epi32(vh, ~m2&m1, vh, vone);
        __mmask16 mx = _mm512_mask_cmpge_epi32_mask(~m2&m1, vh, vh_upper);
        vh = _mm512_mask_mov_epi32(vh, mx, vh_lower);
      }
      vhkey = _mm512_mask_i32gather_epi32(vhkey, ~m2&m1, vh, h_keys, 4);
      m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
      m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);
      aa++;
    }




    __m512i vsquare = _mm512_mullo_epi32(vvalue, vvalue);
    __m512i vsum = vvalue;
    __m512i vcount = vone;
    __m512i vsquaresum = vsquare;

    __m512i vconf = _mm512_conflict_epi32(vh);
    __mmask16 mconf = _mm512_cmpeq_epi32_mask(vconf, vzero);

    __mmask16 mconft = mconf;
   if(mconft != 0xFFFF) { 
      if(!(mconft & 0x2)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+1));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x4)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+2));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x8)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+3));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x10)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+4));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x20)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+5));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x40)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+6));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x80)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+7));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x100)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+8));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x200)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+9));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x400)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+10));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x800)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+11));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }


      if(!(mconft & 0x1000)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+12));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x2000)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+13));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x4000)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+14));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
      if(!(mconft & 0x8000)) {
        overhead++;
        __m512i vt = _mm512_set1_epi32(*((int*)&vh+15));
        __mmask16 mt = _mm512_cmpeq_epi32_mask(vt, vh);
        vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
        vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
        vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
        mconft |= mt;
      }
    }




    if(m1!=65535) {

      _mm512_mask_i32scatter_epi32(h_keys, ~m1&mconf, vh, vkey, 4);
      _mm512_mask_i32scatter_epi32(h_sum, ~m1&mconf, vh, vvalue, 4);
      _mm512_mask_i32scatter_epi32(h_count, ~m1&mconf, vh, vone, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, ~m1&mconf, vh, vsquare, 4);

    }

    if(m2!=0) {

      __m512i vsumt = _mm512_mask_i32gather_epi32(vsum, m1&mconf, vh, h_sum, 4);
      __m512i vcountt = _mm512_mask_i32gather_epi32(vcount, m1&mconf, vh, h_count, 4);
      __m512i vsquaresumt = _mm512_mask_i32gather_epi32(vsquaresum, m1&mconf, vh, h_squaresum, 4);

      vsumt = _mm512_add_epi32(vsumt, vsum);
      vcountt = _mm512_add_epi32(vcountt, vcount);
      vsquaresumt = _mm512_add_epi32(vsquaresumt, vsquaresum);

      _mm512_mask_i32scatter_epi32(h_sum, m1&mconf, vh, vsumt, 4);
      _mm512_mask_i32scatter_epi32(h_count, m1&mconf, vh, vcountt, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, m1&mconf, vh, vsquaresumt, 4);

    }

  }

  //gettimeofday(&tv2, &tz2);
  //cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  //cout << "time used: " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) << endl;

  //print_table();

  //gettimeofday(&tv2, &tz2);
  // cout << "time used: " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) << endl;

  for(int i=0;i<H_SIZE;i+=16) {
    __m512i vh_lower = _mm512_mullo_epi32(_mm512_add_epi32(_mm512_set1_epi32(i), voffset), v16);
    for(int ii=0;ii<15;ii++) {
      __m512i vii = _mm512_add_epi32(vh_lower, _mm512_set1_epi32(ii));  
      __m512i vkeyi = _mm512_i32gather_epi32(vii, h_keys, 4);
      __mmask16 mnone = _mm512_cmpeq_epi32_mask(vkeyi, vnone);
      for(int jj=ii+1;jj<16;jj++) {
        __m512i vjj = _mm512_add_epi32(vh_lower, _mm512_set1_epi32(jj));  
        __m512i vkeyj = _mm512_i32gather_epi32(vjj, h_keys, 4);
        __mmask16 mmerge = _mm512_mask_cmpeq_epi32_mask(~mnone, vkeyi, vkeyj);
        __mmask16 mmove = _mm512_mask_cmpneq_epi32_mask(mnone, vnone, vkeyj);

        if(mmove) {

          _mm512_mask_i32scatter_epi32(h_keys, mmove, vii, vkeyj, 4);
          _mm512_mask_i32scatter_epi32(h_keys, mmove, vjj, vnone, 4);

          __m512i vsumj = _mm512_mask_i32gather_epi32(vzero, mmove, vjj, h_sum, 4);
          __m512i vcountj = _mm512_mask_i32gather_epi32(vzero, mmove, vjj, h_count, 4);
          __m512i vsquaresumj = _mm512_mask_i32gather_epi32(vzero, mmove, vjj, h_squaresum, 4);

          _mm512_mask_i32scatter_epi32(h_sum, mmove, vii, vsumj, 4);
          _mm512_mask_i32scatter_epi32(h_count, mmove, vii, vcountj, 4);
          _mm512_mask_i32scatter_epi32(h_squaresum, mmove, vii, vsquaresumj, 4);

          mnone &= ~mmove;
        }

        if(mmerge) {
          _mm512_mask_i32scatter_epi32(h_keys, mmerge, vjj, vnone, 4);

          __m512i vsumi = _mm512_mask_i32gather_epi32(vzero, mmerge, vii, h_sum, 4);
          __m512i vcounti = _mm512_mask_i32gather_epi32(vzero, mmerge, vii, h_count, 4);
          __m512i vsquaresumi = _mm512_mask_i32gather_epi32(vzero, mmerge, vii, h_squaresum, 4);
          __m512i vsumj = _mm512_mask_i32gather_epi32(vzero, mmerge, vjj, h_sum, 4);
          __m512i vcountj = _mm512_mask_i32gather_epi32(vzero, mmerge, vjj, h_count, 4);
          __m512i vsquaresumj = _mm512_mask_i32gather_epi32(vzero, mmerge, vjj, h_squaresum, 4);

          vsumi = _mm512_add_epi32(vsumi, vsumj);
          vcounti = _mm512_add_epi32(vcounti, vcountj);
          vsquaresumi = _mm512_add_epi32(vsquaresumi, vsquaresumj);


          _mm512_mask_i32scatter_epi32(h_sum, mmerge, vii, vsumi, 4);
          _mm512_mask_i32scatter_epi32(h_count, mmerge, vii, vcounti, 4);
          _mm512_mask_i32scatter_epi32(h_squaresum, mmerge, vii, vsquaresumi, 4);
        }
      }
    }
  }
  gettimeofday(&tv2, &tz2);
  cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  cout << "overhead: " << 16.0 * overhead / num_records << endl;


  int h = my_hash(keys[0])*B;
  while(h_keys[h]!=-1) {
    if(keys[0] == h_keys[h]) {
      cout << h_count[h] << " " << h_sum[h] << " " << h_squaresum[h] << endl;
      break;
    }
    h++;
  }

  return 0;

}
