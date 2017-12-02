#include <tuple>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <cassert>
#include "immintrin.h"
#include <bitset>
using namespace std;

void print_vec(__m512i v) {
  for(int i=0;i<16;i++) {
    cout << *((int *)&v+i) << " ";
  }
  cout << endl;

}

void print_vec(__m512 v) {
  for(int i=0;i<16;i++) {
    cout << *((float *)&v+i) << " ";
  }
  cout << endl;

}

int main(int argc, char *argv[])
{
  int nnodes, nedges;
  ifstream fin(argv[1]); 
  fin >> nnodes >> nedges;

  int *comp = (int *)_mm_malloc(sizeof(int)*nnodes, 64);
  int *comp_new = (int *)_mm_malloc(sizeof(int)*nnodes, 64);
  int *n1 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *n2 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  float *weight = (float *)_mm_malloc(sizeof(float)*nedges, 64);

  // adjancency lists 
  vector<vector<pair<int, float> > > adjs(nnodes);

  for(int i=0;i<nedges;i++) {
    while(fin >> n1[i] >> n2[i] >> weight[i])i++;
  }


  // initial comptances are inf
  for(int i=0;i<nnodes;i++) {
    comp[i] = i;
    comp_new[i] = i;
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

  int nsteps = 0;

  __m512i vzero = _mm512_set1_epi32(0);
  __m512i vnone = _mm512_set1_epi32(-1);
  __m512 vnoneps = _mm512_set1_ps(-1);

  __m512i sindex[65536];
  int nbits[65536];
  for(int i=0;i<65536;i++) {
    int c = 0;
    int t = i;
    int j = 0;
  
    __declspec(align(64)) int temp[16];
    for(int k=0;k<16;k++) temp[k] = 0;
    while(t)
    {
      if(t & 1) {
        temp[j] = c++;
      };
      j++;
      t /= 2;
    }
    nbits[i] = c;
    sindex[i] = _mm512_load_epi32(temp);
  }
  long time = 0;

  __m512i vone = _mm512_set1_epi32(1);


  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  int active_bits = 0, total_bits = 0;

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



    __mmask16 mload= 0xFFFF, mbound;
    __m512i vnx, vny;
    __m512i vdx, vdy;
    

    gettimeofday(&tv1, &tz1);

    int i = 0;
    __m512i vbound = _mm512_set1_epi32(t);

	  while(i<t) {

      __m512i vindex = sindex[mload]; 
      vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(i));
      i += nbits[mload];

      mbound = _mm512_mask_cmplt_epi32_mask(mload, vindex, vbound);
      mload &= mbound;

      vnx = _mm512_mask_i32gather_epi32(vnx, mload, vindex, n1, 4);
      vny = _mm512_mask_i32gather_epi32(vny, mload, vindex, n2, 4);
      vdx = _mm512_mask_i32gather_epi32(vdx, mload, vnx, comp, 4);
      vdy = _mm512_mask_i32gather_epi32(vdy, mload, vny, comp_new, 4);

      __mmask16 mupdate = _mm512_cmpgt_epi32_mask(vdy, vdx);

      __m512i vconf = _mm512_mask_conflict_epi32(vone, mupdate, vny);
      mload = _mm512_mask_cmpeq_epi32_mask(mbound&mupdate, vconf, vzero);

		  _mm512_mask_i32scatter_epi32(comp_new, mload, vny, vdx, 4);
		  _mm512_mask_i32scatter_epi32(wake_list_new, mload, _mm512_add_epi32(_mm512_set1_epi32(ws), sindex[mupdate]), vny, 4);

		  ws += nbits[mload];

      mload |= ~mupdate;

      active_bits += nbits[mload];
      total_bits += 16;
	  }
    
    gettimeofday(&tv2, &tz2);
    time += tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    for(int i=0;i<ws;i++) {
      comp[wake_list_new[i]] = comp_new[wake_list_new[i]];
      wake_list[i] = wake_list_new[i];
    }
    wake_size = ws;

  }



  cout << "time used (microseconds): " << time << endl;
  cout << "nsteps: " << nsteps << endl;
  cout << "comp[5]: " << comp[5] << endl;
  cout << "simd utilization: " << 1.0 * active_bits / total_bits << endl;

  return 0;
}
