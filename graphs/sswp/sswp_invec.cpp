#include <tuple>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <cassert>
#include "immintrin.h"
using namespace std;

long overhead = 0;
inline __m512 invec_reduce(__mmask16 mconf, __m512i vny, __m512 vdy, __mmask16 mupdate) {

  __mmask16 mconft = mconf;
  __m512 vnewdy = vdy;

  if(mconft != 0xFFFF) { 

    if(!(mconft & 0x2) ) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+1));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+2));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+3));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x10)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+4));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x20)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+5));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x40)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+6));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x80)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+7));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x100)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+8));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x200)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+9));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x400)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+10));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x800)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+11));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }


    if(!(mconft & 0x1000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+12));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x2000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+13));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x4000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+14));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
    if(!(mconft & 0x8000)) {
      overhead++;
      __m512i vt = _mm512_set1_epi32(*((int*)&vny+15));
      __mmask16 mt = _mm512_mask_cmpeq_epi32_mask(mupdate, vt, vny);
      vnewdy = _mm512_mask_mov_ps(vnewdy, mt & (~mt+1), _mm512_set1_ps(_mm512_mask_reduce_max_ps(mt, vnewdy)));
      mconft |= mt;
    }
  }

  return vnewdy;


}




int main(int argc, char *argv[])
{
  int nnodes, nedges;
  ifstream fin(argv[1]); 
  fin >> nnodes >> nedges;

  float *width = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  float *width_new = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  int *n1 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *n2 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  float *weight = (float *)_mm_malloc(sizeof(float)*nedges, 64);

  // adjancency lists 
  vector<vector<pair<int, float> > > adjs(nnodes);

  int t = 0;
  while(fin >> n1[t] >> n2[t] >> weight[t])t++;



  __m512i vzero = _mm512_set1_epi32(0);




  // initial widthtances are inf
  width[0] = 99999999;
  for(int i=1;i<nnodes;i++) {
    width[i] = 0;
    width_new[i] = 0;
  }
// init adj list
  for(int i=0;i<nedges;i++) {
    adjs[n1[i]].push_back(make_pair(n2[i], weight[i]));
  }

  int *wake_list = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *wake_list_new = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *waked_nodes = (int *)_mm_malloc(sizeof(int)*nnodes, 64);
  int wake_size = nnodes;
  for(int i=0;i<nnodes;i++) {
      wake_list[i] = i;
  }

  long time = 0;

  int nsteps = 0;

  long total_bits = 0;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;
  // Bellman-Ford
  while(wake_size != 0) {
	  nsteps++;
    int ws = 0;

    for(int i=0;i<nnodes;i++) {
      waked_nodes[i] = 0;
    }

    int t = 0;
    for(int i=0;i<wake_size;i++) {
      if(!waked_nodes[wake_list[i]]) {
        waked_nodes[wake_list[i]] = 1;
        for(auto& aj : adjs[wake_list[i]]) {
          n1[t] = (wake_list[i]);
          n2[t] = (aj.first);
          weight[t] = aj.second;
          t++;
        }
      }
    }

   // cout << t << endl;


    __m512i vnx_next = _mm512_load_epi32(n1);
    __m512i vny_next = _mm512_load_epi32(n2);

    gettimeofday(&tv1, &tz1);
    for(int i=0;i<t/16*16;i+=16) {
      __m512i vnx = vnx_next;
      __m512i vny = vny_next;
      vnx_next = _mm512_load_epi32(n1+i+16);
      vny_next = _mm512_load_epi32(n2+i+16);
      _mm512_prefetch_i32gather_ps(vnx_next, width, 4, _MM_HINT_T0);
      _mm512_prefetch_i32gather_ps(vny_next, width, 4, _MM_HINT_T0);
      __m512 vw = _mm512_load_ps(weight+i);
      __m512 vdx = _mm512_i32gather_ps(vnx, width, 4);
      __m512 vdy = _mm512_i32gather_ps(vny, width, 4);
      __m512 vnewdy = _mm512_min_ps(vdx, vw);
      __mmask16 mupdate = _mm512_cmp_ps_mask(vdy, vnewdy, _CMP_LT_OS);
       __m512i vconf = _mm512_maskz_conflict_epi32(mupdate, vny);

      __mmask16 mconf = _mm512_cmpeq_epi32_mask(vconf, vzero);

      vnewdy = invec_reduce(mconf, vny, vnewdy, mupdate);
      total_bits += 16;

      _mm512_mask_i32scatter_ps(width_new, mconf&mupdate, vny, vnewdy, 4);
      _mm512_store_epi32(wake_list_new+ws, _mm512_maskz_compress_epi32(mconf&mupdate, vny));
      ws += _popcnt32(mupdate&mconf);
    }


    for(int i=t/16*16;i<t;i++) {
      int nx = n1[i];
      int ny = n2[i];
      float dx = width[nx];
      float dy = width[ny];
      float w = weight[i];

      float m = dx < w ? dx : w;
      if(dy < m) {
        width[ny] = m;
        wake_list_new[ws++] = ny;
      }
    }

  gettimeofday(&tv2, &tz2);
  time +=   tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    for(int i=0;i<ws;i++) {
    width[wake_list_new[i]] = width_new[wake_list_new[i]];
      wake_list[i] = wake_list_new[i];
    }
    wake_size = ws;
  }



  cout << "time used (microseconds): " << time << endl;
  cout << "nsteps: " << nsteps << endl;
  cout << "width[5]: " << width[5] << endl;
  cout << "overhead: " << 16.0*overhead/total_bits << endl;

  return 0;
}
