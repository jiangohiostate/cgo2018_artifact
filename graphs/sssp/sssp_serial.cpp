#include <tuple>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <cassert>
using namespace std;

int main(int argc, char *argv[])
{
  int nnodes, nedges;
  ifstream fin(argv[1]); 
  fin >> nnodes >> nedges;

  float *dis = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  float *dis_new = (float *)_mm_malloc(sizeof(float)*nnodes, 64);
  int *n1 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  int *n2 = (int *)_mm_malloc(sizeof(int)*nedges, 64);
  float *weight = (float *)_mm_malloc(sizeof(float)*nedges, 64);

  // adjancency lists 
  vector<vector<pair<int, float> > > adjs(nnodes);

  int t = 0;
  while(fin >> n1[t] >> n2[t] >> weight[t])t++;
  assert(t==nedges);

  // initial distances are inf
  dis[0] = 0;
  for(int i=1;i<nnodes;i++) {
    dis[i] = 99999999;
    dis_new[i] = 99999999;
  }
// init adj list
  for(int i=0;i<nedges;i++) {
    adjs[n1[i]].push_back(make_pair(n2[i], weight[i]));
  }


  int *wake_list = (int *)_mm_malloc(sizeof(int)*max(nedges,nnodes), 64);
  int *wake_list_new = (int *)_mm_malloc(sizeof(int)*max(nedges,nnodes), 64);
  int *waked_nodes = (int *)_mm_malloc(sizeof(int)*nnodes, 64);
  int wake_size = nnodes;

for(int i=0;i<nnodes;i++) {
      wake_list[i] = i;
  }

  int nsteps = 0;
  long time = 0;


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
          n1[t] = wake_list[i];
          n2[t] = aj.first;
          weight[t] = aj.second;
          t++;
        }
      }
    }

  gettimeofday(&tv1, &tz1);
    for(int i=0;i<t;i++) {
      int nx = n1[i];
      int ny = n2[i];
      float dx = dis[nx];
      float w = weight[i];

      if(dis_new[ny] > dx + w) {
        dis_new[ny] = dx + w;
        wake_list_new[ws++] = ny;
      }
    }

  gettimeofday(&tv2, &tz2);
  time += tv2.tv_usec - tv1.tv_usec + 1000000 * (tv2.tv_sec - tv1.tv_sec); 

    for(int i=0;i<ws;i++) {
      dis[wake_list_new[i]] = dis_new[wake_list_new[i]];
      wake_list[i] = wake_list_new[i];
    }

    wake_size = ws;

  }

  cout << "time used (microseconds): " << time << endl;
  cout << "nsteps: " << nsteps << endl;
  cout << "dis[5]: " << dis[5] << endl;

  return 0;
}
