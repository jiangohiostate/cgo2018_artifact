#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <tuple>
#include "immintrin.h"
#include <cassert>
using namespace std;

#define DUMP 0.85
#define MAX_NODES 1700000
#define MAX_EDGES 40000000

void print_vec(__m512 v) {
  for(int i=0;i<16;i++) {
    cout << *((float *)&v+i) << " ";
  }
  cout << endl;
}

void print_vec(__m512i v) {
  for(int i=0;i<16;i++) {
    cout << *((int *)&v+i) << " ";
  }
  cout << endl;
}

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

  struct timeval tv1, tv2;
  struct timezone tz1, tz2;
  // tilling

#define TILESIZE 32768

  gettimeofday(&tv1, &tz1);

  int dtiles = nnodes / TILESIZE;
  if(nnodes % TILESIZE)dtiles++;
  int ntiles = dtiles * dtiles;
  vector<vector<pair<int, int> > > tiles(ntiles);
  for(int i=0;i<nedges;i++) {
    int tx = n1[i] / TILESIZE;
    int ty = n2[i] / TILESIZE;
    tiles[tx*dtiles+ty].push_back(make_pair(n1[i], n2[i]));
  }
  gettimeofday(&tv2, &tz2);

  cout << "tiling time (ms): " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) << endl;

  cout << "nedges: " << nedges << endl;

  nedges = 0;


  gettimeofday(&tv1, &tz1);
  vector<vector<vector<pair<int, int>> > > tiles_of_groups(ntiles);
  
  for(int i=0;i<ntiles;i++) {
      vector<vector<pair<int, int> > > groups;
      for(auto& edge : tiles[i]) {
        bool inserted_flag = false;
        for(auto& g : groups) {
          if(g.size() < 16) {
            bool in_flag = false;
            for(auto& e : g) {
              if(e.second == edge.second) {
                in_flag = true;
                break;
              }
            }
            if(!in_flag) {
              g.push_back(edge);
              inserted_flag = true;
              break;
            }
          }
        }
        if(!inserted_flag) {
          vector<pair<int, int> > ng;
          ng.push_back(edge);
          groups.push_back(ng);
        }
      }
      tiles_of_groups.push_back(groups);
      for(auto& g : groups) {
        nedges += g.size();
        while(nedges % 16) {
          nedges++;
        }
      }
  }


  _mm_free(n1);
  _mm_free(n2);

  n1 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  n2 = (int *)_mm_malloc(sizeof(int)*nedges, 64);

  t = 0;
  for(auto& groups : tiles_of_groups) {
    for(auto& g : groups) {
      for(auto& e : g) {
        n1[t] = e.first;
        n2[t] = e.second;
        t++;
      }
      while(t%16) {
        n1[t] = -1;
        t++;
      }
    }
  }

  assert(t==nedges);

  gettimeofday(&tv2, &tz2);
  cout << "grouping time (ms): " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) << endl;

  cout << "reordered nedges: " << nedges << endl;

  float *rank = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  float *sum = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  __m512i vnone = _mm512_set1_epi32(-1);
  __m512 vdumb = _mm512_set1_ps(0);


  long time = 0;



  for(int i=0;i<nnodes;i++) {
    rank[i] = 1.0;
    sum[i] = 0.0;
  }

  int step = 0;
  float last_rank;

  do {
    step++;
    last_rank = rank[100];
    for(int i=0;i<nnodes;i++) {
      sum[i] = 0.0;
    }

    gettimeofday(&tv1, &tz1);

    for(int i=0;i<nedges;i+=16) {
      __m512i vnx = _mm512_load_epi32(n1+i);
      __m512i vny = _mm512_load_epi32(n2+i);
      __mmask16 m = _mm512_cmpneq_epi32_mask(vnx, vnone);
      __m512 vrankx = _mm512_mask_i32gather_ps(vdumb, m, vnx, rank, 4);
      __m512 vnnx = _mm512_mask_i32gather_ps(vdumb, m, vnx, nneibor, 4);
      __m512 vsumy = _mm512_mask_i32gather_ps(vdumb, m, vny, sum, 4);
      __m512 vadd = _mm512_mask_div_ps(vdumb, m, vrankx, vnnx);
      vsumy = _mm512_add_ps(vsumy, vadd);
      _mm512_mask_i32scatter_ps(sum, m, vny, vsumy, 4);
    }

    gettimeofday(&tv2, &tz2);

    time += tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    for(int i = 0; i < nnodes; i++)
    {
      rank[i] = (1 - DUMP) / nnodes + DUMP * sum[i]; 	
    }


  } while(((rank[100] - last_rank) > 0.001*last_rank) || ((rank[100]-last_rank) < -1*0.001*last_rank));


  cout << "rank[100]: " << rank[100] << endl;
  cout << "total computing time: " << time / 1000000.0 << " sec" << endl;
  cout << "number of steps: " << step << endl;

  return 0;
}
