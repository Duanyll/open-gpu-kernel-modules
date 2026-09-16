// Prove P2P writes land correctly in framebuffer ABOVE the 32 GB BAR1 boundary
// on a 48 GB-modded RTX 4090 -- the exact case ChihayaK reported as failing in
// tinygrad open-gpu-kernel-modules #36 ("writes above 32 GB fail", concluding a
// 64 GB-BAR VBIOS was required).
//
// Method: allocate ONE large buffer on the destination GPU (default 34 GiB, i.e.
// just past the 32 GiB BAR1 size), then P2P-write a position-keyed pattern into
// only its TAIL (last 512 MiB, whose physical pages sit above 32 GiB) from a peer
// GPU and verify. The live peer-mapped working set is only the tail, well within
// BAR1. A clean PASS means Method 3's dynamic per-allocation BAR1 window maps a
// small aperture onto high framebuffer pages correctly -- which the static scheme
// (bus_addr = BAR1_base + fb_phys_offset) cannot do for offsets > 32 GiB.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); fflush(stdout); exit(1);} } while(0)

__global__ void fill(uint64_t* p, size_t n, uint64_t base){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) p[i] = (base + i)*0x9E3779B97F4A7C15ull;
}
__global__ void check(const uint64_t* p, size_t n, uint64_t base, unsigned long long* bad){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n){ uint64_t exp=(base+i)*0x9E3779B97F4A7C15ull; if(p[i]!=exp) atomicAdd(bad,1ull); }
}

int main(int argc,char**argv){
  int src     = argc>1 ? atoi(argv[1]) : 0;
  int dst     = argc>2 ? atoi(argv[2]) : 1;
  double bigG = argc>3 ? atof(argv[3]) : 34.0;              // single dst allocation size
  size_t tailMiB = argc>4 ? (size_t)atoll(argv[4]) : 512;   // tail region we P2P-write
  size_t tailBytes = tailMiB*1024*1024;

  int can=0; CK(cudaDeviceCanAccessPeer(&can,src,dst));
  printf("src=GPU%d dst=GPU%d  bigBuf=%.1f GiB  tail=%zu MiB  cudaDeviceCanAccessPeer=%d\n",
         src,dst,bigG,tailMiB,can);
  if(!can){ printf("NO P2P between this pair -- abort\n"); return 2; }
  CK(cudaSetDevice(src));
  cudaError_t pe=cudaDeviceEnablePeerAccess(dst,0);
  if(pe!=cudaSuccess && pe!=cudaErrorPeerAccessAlreadyEnabled){
    printf("cudaDeviceEnablePeerAccess FAIL: %s\n",cudaGetErrorString(pe)); return 1; }

  CK(cudaSetDevice(dst));
  size_t freeB=0,totB=0; CK(cudaMemGetInfo(&freeB,&totB));
  printf("dst GPU%d FB: total=%.1f GiB free=%.1f GiB\n",dst,totB/1073741824.0,freeB/1073741824.0);

  // Largest single allocation that fits, starting from the requested size.
  size_t want=(size_t)(bigG*1073741824.0);
  uint64_t* d=nullptr; cudaError_t me;
  while((me=cudaMalloc(&d,want))!=cudaSuccess && want>(size_t)2*1073741824){
    cudaGetLastError(); want -= (size_t)1*1073741824;   // back off 1 GiB and retry
  }
  if(me!=cudaSuccess){ printf("could not allocate a large single buffer: %s\n",cudaGetErrorString(me)); return 1; }
  printf("single dst allocation = %.1f GiB\n", want/1073741824.0);
  if(want <= (size_t)32*1073741824)
    printf("WARNING: allocation <= 32 GiB, tail is NOT above the BAR1 boundary\n");

  size_t off = want - tailBytes;   // tail offset; physical page here is > 32 GiB when want > 32 GiB
  printf("writing to tail at byte offset %.2f GiB (physical page > 32 GiB)\n", off/1073741824.0);

  unsigned long long* dbad=nullptr; CK(cudaMalloc(&dbad,8));
  CK(cudaSetDevice(src));
  uint64_t* s=nullptr; CK(cudaMalloc(&s,tailBytes));
  const size_t n=tailBytes/8;
  fill<<<(n+255)/256,256>>>(s,n,0xABCDEFull); CK(cudaDeviceSynchronize());

  CK(cudaMemcpyPeer((char*)d+off,dst,s,src,tailBytes));   // P2P write into the HIGH tail
  CK(cudaSetDevice(dst)); CK(cudaDeviceSynchronize());
  CK(cudaMemset(dbad,0,8));
  check<<<(n+255)/256,256>>>((uint64_t*)((char*)d+off),n,0xABCDEFull,dbad); CK(cudaDeviceSynchronize());
  unsigned long long bad=0; CK(cudaMemcpy(&bad,dbad,8,cudaMemcpyDeviceToHost));

  cudaEvent_t a,b; CK(cudaSetDevice(src)); CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  for(int t=0;t<20;t++) CK(cudaMemcpyPeer((char*)d+off,dst,s,src,tailBytes));
  CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
  float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
  double gbps=(double)tailBytes*20/(ms/1e3)/1e9;

  printf("P2P write to >32 GiB tail: mismatches=%llu, peer-write BW=%.1f GB/s -> %s\n",
         bad, gbps, bad?"FAIL":"PASS");
  return bad?1:0;
}
