// Does enabling P2P peer access cap usable VRAM to ~BAR1 (32 GiB), or is all
// 48 GiB still allocatable? Enables peer access src<->dst, then allocates 256 MiB
// chunks on dst until failure and reports the total. Compare against the no-P2P
// baseline (~46.75 GiB). Then P2P-writes+verifies a chunk near the TOP of what was
// allocated (physical > 32 GiB) to confirm high pages are peer-accessible.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); fflush(stdout); exit(1);} } while(0)

__global__ void fill(uint64_t* p, size_t n, uint64_t base){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if(i<n) p[i]=(base+i)*0x9E3779B97F4A7C15ull; }
__global__ void check(const uint64_t* p, size_t n, uint64_t base, unsigned long long* bad){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
  if(i<n){ uint64_t e=(base+i)*0x9E3779B97F4A7C15ull; if(p[i]!=e) atomicAdd(bad,1ull); } }

int main(int argc,char**argv){
  int src=argc>1?atoi(argv[1]):0;
  int dst=argc>2?atoi(argv[2]):1;
  size_t chunkMiB=argc>3?(size_t)atoll(argv[3]):256;
  size_t chunk=chunkMiB*1024*1024;

  int can=0; CK(cudaDeviceCanAccessPeer(&can,src,dst));
  CK(cudaSetDevice(src)); cudaError_t e1=cudaDeviceEnablePeerAccess(dst,0);
  CK(cudaSetDevice(dst)); cudaError_t e2=cudaDeviceEnablePeerAccess(src,0);
  printf("peer access enabled: src->dst=%d dst->src=%d canAccessPeer=%d\n",
         (int)e1,(int)e2,can);
  if(e1!=cudaSuccess&&e1!=cudaErrorPeerAccessAlreadyEnabled){printf("enable FAIL\n");return 1;}

  size_t freeB=0,totB=0; CK(cudaSetDevice(dst)); CK(cudaMemGetInfo(&freeB,&totB));
  printf("dst GPU%d free=%.2f total=%.2f GiB\n",dst,freeB/1073741824.0,totB/1073741824.0);

  std::vector<void*> v; size_t tot=0;
  while(true){ void*p=nullptr; if(cudaMalloc(&p,chunk)!=cudaSuccess){cudaGetLastError();break;} v.push_back(p); tot+=chunk; }
  printf("allocated %.2f GiB on dst GPU%d WITH peer access enabled (%zu x %zu MiB chunks)\n",
         tot/1073741824.0,dst,v.size(),chunkMiB);
  if(v.empty()){printf("nothing allocated\n");return 1;}

  // P2P-write+verify a chunk near the TOP of what we allocated (physical > 32 GiB
  // if total > 32 GiB). Pick the last successfully allocated chunk.
  uint64_t* dtop=(uint64_t*)v.back();
  double topFloorGiB = (tot - chunk)/1073741824.0;
  size_t n=chunk/8;
  unsigned long long* dbad=nullptr; CK(cudaMalloc(&dbad,8)); CK(cudaMemset(dbad,0,8));
  CK(cudaSetDevice(src)); uint64_t* s=nullptr; CK(cudaMalloc(&s,chunk));
  fill<<<(n+255)/256,256>>>(s,n,0x1234ull); CK(cudaDeviceSynchronize());
  CK(cudaMemcpyPeer(dtop,dst,s,src,chunk));
  CK(cudaSetDevice(dst)); CK(cudaDeviceSynchronize());
  check<<<(n+255)/256,256>>>(dtop,n,0x1234ull,dbad); CK(cudaDeviceSynchronize());
  unsigned long long bad=0; CK(cudaMemcpy(&bad,dbad,8,cudaMemcpyDeviceToHost));
  printf("P2P write+verify to the highest allocated chunk (physical floor ~%.1f GiB): mismatches=%llu -> %s\n",
         topFloorGiB, bad, bad?"FAIL":"PASS");
  return bad?1:0;
}
