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
  int *nneibor = (int *)_mm_malloc(sizeof(int)*nnodes, 64);

  for(int i=0;i<nnodes;i++) {
    nneibor[i] = 0;
  }

  int t = 0;
  int w;
  while(fin >> n1[t] >> n2[t] >> w) {
    nneibor[n1[t]]++;
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

  t = 0;
  for(int i=0;i<ntiles;i++) {
      for(auto& edge : tiles[i]) {
        n1[t] = edge.first;
        n2[t] = edge.second;
        t++;
      }
  }
  gettimeofday(&tv2, &tz2);
  cout << "tiling time: " << tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec) <<  " ms"<< endl;


  float *rank = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  float *sum = (float *)_mm_malloc(sizeof(float)*nnodes, 64);

  for(int i=0;i<nnodes;i++) {
    rank[i] = 1.0;
    sum[i] = 0.0;
  }
  long time = 0;

  int step = 0;
  float last_rank;

  do {
    last_rank = rank[100];

    step++;


    for(int i=0;i<nnodes;i++) {
      sum[i] = 0.0;
    }

    gettimeofday(&tv1, &tz1);

    for(int j=0;j<nedges;j++) {
      int nx = n1[j];
      int ny = n2[j];
      sum[ny] += rank[nx] / nneibor[nx];
    }

    gettimeofday(&tv2, &tz2);

    time += tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    for(int j = 0; j < nnodes; j++)
    {
      rank[j] = (1 - DUMP) / nnodes + DUMP * sum[j]; 	
    }

    // cout << last_rank << " " << rank[100] << endl;

  } while(((rank[100] - last_rank) > 0.001*last_rank) || ((rank[100]-last_rank) < -1*0.001*last_rank));


  cout << "rank[100]: " << rank[100] << endl;
  cout << "total computing time: " << time / 1000000.0 << " sec" << endl;
  cout << "number of steps: " << step << endl;

  return 0;
}
