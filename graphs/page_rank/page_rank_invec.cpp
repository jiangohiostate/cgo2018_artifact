#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include "immintrin.h"
#include "sse_api.h"
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



  for(int i=0;i<nnodes;i++) {
	  rank[i] = 1.0;
	  sum[i] = 0.0;
  }

  long time = 0;

  vfloat vzero = 0.0;

  long total_bits = 0;

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

	  gettimeofday(&tv1, &tz1);



	  for(int i=0;i<nedges/16*16;i+=16) {
                  vint vnx, vny;
                  vnx.load(n1+i);
                  vny.load(n2+i);

                  vfloat vrankx, vnnx, vsumy, vadd;
                  vrankx.load(rank, vnx, 4);
                  vnnx.load(nneibor, vnx, 4);
                  vsumy.load(sum, vny, 4);
                  vadd = vrankx / vnnx;

                  mask m = invec_add(0xFFFF, vny, vadd);
                  vsumy += vadd;

                  Mask::set_mask(m, vzero);
                  vsumy.mask().store(sum, vny);

                  total_bits += 16;

	  }

	  for(int j=nedges/16*16;j<nedges;j++) {
		  int nx = n1[j];
		  int ny = n2[j];
		  sum[ny] += rank[nx] / nneibor[nx];
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
