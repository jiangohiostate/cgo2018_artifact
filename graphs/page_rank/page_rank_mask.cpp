#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include "immintrin.h"
#include <cassert>
using namespace std;

#define DUMP 0.85
#define MAX_NODES 1700000
#define MAX_EDGES 40000000


int nnodes, nedges;

int main(int argc, char *argv[]) {

	ifstream fin(argv[1]);
	fin >>  nnodes >> nedges;

  int *n1 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *n2 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  float *nneibor = (float *)_mm_malloc(sizeof(float)*nnodes, 64);

  for(int i=0;i<nnodes;i++) {
    nneibor[i] = 0;
  }

  int t = 0;
  int w;
  while(fin >> n1[t] >> n2[t] >> w) {
    nneibor[n1[t]] += 1.0;
    t++;
  }
  assert(t==nedges);
  cout << "input complete." << endl;

  // tilling

#define TILESIZE 32768

  int dtiles = nnodes / TILESIZE;
  if(nnodes % TILESIZE)dtiles++;
  int ntiles = dtiles * dtiles;
  vector<vector<pair<int, int> > > tiles(ntiles);
  for(int i=0;i<nedges;i++) {
    int tx = n1[i] / TILESIZE;
    int ty = n2[i] / TILESIZE;
    tiles[tx*dtiles+ty].push_back(make_pair(n1[i], n2[i]));
  }

  t = 0;

  for(int i=0;i<ntiles;i++) {
      for(auto& edge : tiles[i]) {
        n1[t] = edge.first;
        n2[t] = edge.second;
        t++;
      }
    }



  float *rank = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  float *sum = (float *)_mm_malloc(sizeof(float)*nnodes, 64);

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


  __m512i vzero = _mm512_set1_epi32(0);
  __m512i vbound = _mm512_set1_epi32(nedges);





  for(int i=0;i<nnodes;i++) {
    rank[i] = 1.0;
    sum[i] = 0.0;
  }


  long active_bits = 0, total_bits = 0;
  long time = 0;

  int step = 0;
  float last_rank;

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;

  do {
    step++;
    last_rank = rank[100];

    for(int i=0;i<nnodes;i++) {
      sum[i] = 0.0;
    }

    __mmask16 mload = 0xFFFF;
    __m512i vnx, vny;
    __m512 vrankx, vnnx, vsumy;

    gettimeofday(&tv1, &tz1);

    int i = 0;

    while(i<nedges) {
      __m512i vindex = sindex[mload]; 
      vindex = _mm512_add_epi32(vindex, _mm512_set1_epi32(i));
      i += nbits[mload];

      __mmask16 mbound = _mm512_mask_cmplt_epi32_mask(mload, vindex, vbound);
      mload &= mbound;

      vnx = _mm512_mask_i32gather_epi32(vnx, mload, vindex, n1, 4);
      vny = _mm512_mask_i32gather_epi32(vny, mload, vindex, n2, 4);
      vrankx = _mm512_mask_i32gather_ps(vrankx, mload, vnx, rank, 4);
      vnnx = _mm512_mask_i32gather_ps(vnnx, mload, vnx, nneibor, 4);


      __m512i vconf = _mm512_conflict_epi32(vny);
      mload = _mm512_cmpeq_epi32_mask(vconf, vzero);

      total_bits += 16;

      vsumy = _mm512_mask_i32gather_ps(vsumy, mload, vny, sum, 4);
      _mm512_mask_i32scatter_ps(sum, mload, vny, _mm512_add_ps(vsumy, _mm512_div_ps(vrankx, vnnx)), 4);
    }

    gettimeofday(&tv2, &tz2);

    time += tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    for(int i = 0; i < nnodes; i++)
    {
      rank[i] = (1 - DUMP) / nnodes + DUMP * sum[i]; 	
    }

  } while(((rank[100] - last_rank) > 0.001*last_rank) || ((rank[100]-last_rank) < -1*0.001*last_rank));


  cout << "simd utilization: " << 1.0 * nedges * step / total_bits << endl;
  cout << "rank[100]: " << rank[100] << endl;
  cout << "total computing time: " << time / 1000000.0 << " sec" << endl;
  cout << "number of steps: " << step << endl;

  return 0;
}
