#include <iostream>
#include <fstream>
#include <sys/time.h>
#include "immintrin.h"
#include <stdlib.h>
using namespace std;

int *keys;
int *values;


#define T 2654435769 

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



  cout << "input finish! " << endl;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  gettimeofday(&tv1, &tz1);
  //struct timeval tv1, tv2;
 // struct timezone tz1, tz2;

 // gettimeofday(&tv1, &tz1);

  __mmask16 mload = 0xFFFF;
  __m512i vkey = vzero;
  __m512i vvalue = vzero;
  __m512i vh;
  int i = 0;

  int niter = 0;


  while(i<num_records) {
    niter++;

    __m512i vindex = gather_indices[mload];
    vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(i));

  //  mload = _mm512_mask_cmplt_epi32_mask(mload, vindex, vbound);

    vkey = _mm512_mask_i32gather_epi32(vkey, mload, vindex, keys, 4);
    vvalue = _mm512_mask_i32gather_epi32(vvalue, mload, vindex, values, 4);
    i += num_bits[mload];

    __m512i vh1 = v_my_hash(vkey);
    vh1 = _mm512_mullo_epi32(vh1, _mm512_set1_epi32(B));

    __m512i vh_lower = vh1;
    __m512i vh_upper = _mm512_add_epi32(vh1, v16);

    __declspec(align(64)) int lower_temp[16];
   _mm512_store_epi32(lower_temp, vh1);

    vh = _mm512_mask_add_epi32(vh, mload, vh1, vshuffle);

    //_mm512_prefetch_i32scatter_ps(h_sum, vh, 4, _MM_HINT_T0);
   // _mm512_prefetch_i32scatter_ps(h_count, vh, 4, _MM_HINT_T0);
   // _mm512_prefetch_i32scatter_ps(h_squaresum, vh, 4, _MM_HINT_T0);

    __m512i vhkey = _mm512_i32gather_epi32(vh, h_keys, 4);



    __mmask16 m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
    __mmask16 m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);

    int aa = 0;
    while(m1 != m2) {
      if(aa==16) {

        //print_table();

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


    _mm512_i32scatter_epi32(h_keys, vh, voffset, 4);
    __m512i vtest = _mm512_i32gather_epi32(vh, h_keys, 4);
    __mmask16 mnoconf = _mm512_cmpeq_epi32_mask(vtest, voffset);
    _mm512_i32scatter_epi32(h_keys, vh, vhkey, 4);
 //   __m512i vr = _mm512_conflict_epi32(vh);
//  __mmask16 mnoconf = _mm512_cmpeq_epi32_mask(vr, vzero);
    
    mload = mnoconf;


      __m512i vsquare = _mm512_mullo_epi32(vvalue, vvalue);
      __m512i vsum = vvalue;
      __m512i vcount = vone;
      __m512i vsquaresum = vsquare;

      if(m1!=65535) {

        _mm512_mask_i32scatter_epi32(h_keys, ~m1&mnoconf, vh, vkey, 4);
        _mm512_mask_i32scatter_epi32(h_sum, ~m1&mnoconf, vh, vvalue, 4);
        _mm512_mask_i32scatter_epi32(h_count, ~m1&mnoconf, vh, vone, 4);
        _mm512_mask_i32scatter_epi32(h_squaresum, ~m1&mnoconf, vh, vsquare, 4);

      }

      if(m2!=0) {
        vsum = _mm512_mask_i32gather_epi32(vsum, m2&mnoconf, vh, h_sum, 4);
        vcount = _mm512_mask_i32gather_epi32(vcount, m2&mnoconf, vh, h_count, 4);
        vsquaresum = _mm512_mask_i32gather_epi32(vsquaresum, m2&mnoconf, vh, h_squaresum, 4);

        vsum = _mm512_add_epi32(vsum, vvalue);
        vcount = _mm512_add_epi32(vcount, vone);
        vsquaresum = _mm512_add_epi32(vsquaresum, vsquare);

        _mm512_mask_i32scatter_epi32(h_sum, m2&mnoconf, vh, vsum, 4);
        _mm512_mask_i32scatter_epi32(h_count, m2&mnoconf, vh, vcount, 4);
        _mm512_mask_i32scatter_epi32(h_squaresum, m2&mnoconf, vh, vsquaresum, 4);
      }

  }


  //gettimeofday(&tv2, &tz2);
  //cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  //cout << "time used: " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) << endl;

  for(int i=0;i<16;i++) {
    if(~mload & (1<<i)) {
      int key = *((int *)(&vkey)+i);
      int value = *((int *)(&vvalue)+i);
      int h = my_hash(key)*B;
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
        if(h==h/16*16+16)h = h/16*16;
      }
      if(!flag) {
        h_keys[h] = key;
        h_sum[h] = value;
        h_count[h] = 1;
        h_squaresum[h] = value * value;
      }
    }
  }


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
  cout << "simd utilization: " << num_records / (niter*16.0) << endl;
  cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  

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
