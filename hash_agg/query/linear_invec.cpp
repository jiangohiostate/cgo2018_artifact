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
  h_count = (int *)_mm_malloc(sizeof(int)*2*H_SIZE, 64);
  h_sum = (int *)_mm_malloc(sizeof(int)*2*H_SIZE, 64);
  h_squaresum = (int *)_mm_malloc(sizeof(int)*2*H_SIZE, 64);

  for(int i=0;i<H_SIZE;i++) {
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


  cout << "input finish! " << endl;

  __m512i voffset = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
  __m512i vnone = _mm512_set1_epi32(-1);
  __m512i vntwo = _mm512_set1_epi32(-2);
  __m512i vone = _mm512_set1_epi32(1);
  __m512i vzero = _mm512_set1_epi32(0);
  __m512i vupper = _mm512_set1_epi32(H_SIZE);

  __m512i vbound = _mm512_set1_epi32(cur);

  int overhead = 0;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  gettimeofday(&tv1, &tz1);

  for(int i=0; i<cur; i+=16) {

  __m512i vkey = _mm512_load_epi32(keys+i);
  __m512i vvalue = _mm512_load_epi32(values+i);
  __m512i vh = v_my_hash(vkey);

  //cout << "==============" << endl;
  //print_vec(vh);
 // cout << "-------------" << endl;


    //_mm512_prefetch_i32gather_ps(vh_next, h_keys, 4, _MM_HINT_T0);
   // _mm512_prefetch_i32gather_ps(h_count, vh_next, 4, _MM_HINT_T0);
   // _mm512_prefetch_i32gather_ps(h_squaresum, vh_next, 4, _MM_HINT_T0);

    __m512i vhkey = _mm512_i32gather_epi32(vh, h_keys, 4);

    __mmask16 m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
    __mmask16 m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);

    while(m1 != m2) {
      vh = _mm512_mask_add_epi32(vh, ~m2&m1, vh, vone);
      __mmask mx = _mm512_mask_cmpgt_epi32_mask(~m2&m1, vh, vupper);
      vh = _mm512_mask_mov_epi32(vh, mx, vzero); 
      vhkey = _mm512_mask_i32gather_epi32(vhkey, ~m2&m1, vh, h_keys, 4);
      m1 = _mm512_cmpneq_epi32_mask(vhkey, vnone);
      m2 = _mm512_mask_cmpeq_epi32_mask(m1, vhkey, vkey);
    }

    __m512i vsquare = _mm512_mullo_epi32(vvalue, vvalue);

    __m512i vconf = _mm512_conflict_epi32(vh);
    __mmask16 mconf = _mm512_cmpeq_epi32_mask(vconf, vzero);
    __m512i vconf2 = _mm512_conflict_epi32(_mm512_mask_mov_epi32(vh, mconf, vntwo));
    __mmask16 mconf2 = ~mconf & _mm512_cmpeq_epi32_mask(vconf2, vzero);

    __m512i vsum = vvalue;
    __m512i vcount = vone;
    __m512i vsquaresum = vsquare;

//   cout << i << endl;
    __mmask16 mconft = mconf | mconf2;
    __mmask16 mconff = mconft;

  if(mconft != 0xFFFF) { 
    if(!(mconft & 0x2)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+1));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x4)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+2));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x8)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+3));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x10)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+4));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x20)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+5));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x40)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+6));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x80)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+7));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x100)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+8));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x200)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+9));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x400)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+10));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x800)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+11));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }


    if(!(mconft & 0x1000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+12));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x2000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+13));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x4000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+14));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
    if(!(mconft & 0x8000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vh+15));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(~mconf2, vt, vh);
      vcount = _mm512_mask_mov_epi32(vcount, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vcount)));
      vsum = _mm512_mask_mov_epi32(vsum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsum)));
      vsquaresum = _mm512_mask_mov_epi32(vsquaresum, mt & (~mt+1), _mm512_set1_epi32(_mm512_mask_reduce_add_epi32(mt, vsquaresum)));
      mconft |= mt;
    }
  }



//cout << "i: " << i << endl;

  //  cout << mconf << endl;

  // print_vec(vh);
  // print_vec(vsum);
  //  print_vec(vcount);
  // print_vec(vsquaresum);

   // cout << "**********" << endl;
   //
   __m512i vhh = _mm512_slli_epi32(vh, 1);
   vhh = _mm512_mask_add_epi32(vhh, mconf2, vone, vhh);


    if(m1!=65535) {

      _mm512_mask_i32scatter_epi32(h_keys, ~m1&mconf, vh, vkey, 4);
      _mm512_mask_i32scatter_epi32(h_sum, ~m1&(mconff), vhh, vsum, 4);
      _mm512_mask_i32scatter_epi32(h_count, ~m1&(mconff), vhh, vcount, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, ~m1&(mconff), vhh, vsquaresum, 4);
    }

    if(m1!=0) {
      __m512i vsumt = _mm512_mask_i32gather_epi32(vsum, m1&mconff, vhh, h_sum, 4);
      __m512i vcountt = _mm512_mask_i32gather_epi32(vcount, m1&mconff, vhh, h_count, 4);
      __m512i vsquaresumt = _mm512_mask_i32gather_epi32(vsquaresum, m1&mconff, vhh, h_squaresum, 4);

      vsumt = _mm512_add_epi32(vsumt, vsum);
      vcountt = _mm512_add_epi32(vcountt, vcount);
      vsquaresumt = _mm512_add_epi32(vsquaresumt, vsquaresum);

      _mm512_mask_i32scatter_epi32(h_sum, m1&mconff, vhh, vsumt, 4);
      _mm512_mask_i32scatter_epi32(h_count, m1&mconff, vhh, vcountt, 4);
      _mm512_mask_i32scatter_epi32(h_squaresum, m1&mconff, vhh, vsquaresumt, 4);
    }

  }

  for(int i=0;i<H_SIZE;i++) {
    h_sum[2*i] += h_sum[2*i+1];
    h_count[2*i] += h_count[2*i+1];
    h_squaresum[2*i] += h_squaresum[2*i+1];
  }




  gettimeofday(&tv2, &tz2);
  cout << "throughput: " << 32*1024*1024*1.0 / (tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec)) << endl;
  cout << "overhead: " << 16.0 * overhead / cur << endl;

  int h = my_hash(keys[0]);
  while(h_keys[h]!=-1) {
    if(keys[0] == h_keys[h]) {
      cout << h_count[2*h] << " " << h_sum[2*h] << " " << h_squaresum[2*h] << endl;
      break;
    }
    h++;
    if(h==H_SIZE)h = 0;
  }
  return 0;

}
