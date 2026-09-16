// Minimal P2P correctness + bandwidth check for dynamic BAR1 P2P.
// Tests all ordered GPU pairs visible to the process: enables peer access,
// fills src with an index-derived pattern, cudaMemcpyPeer -> dst, copies dst
// back to host, verifies every element, then times the peer copy.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void fill(uint32_t* p, size_t n, uint32_t seed){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  if(i<n) p[i] = (uint32_t)(i*2654435761u + seed);
}

int main(int argc, char** argv){
  size_t MB = (argc>1)? atoll(argv[1]) : 256;
  size_t bytes = MB*1024*1024;
  size_t n = bytes/4;
  int nd; CK(cudaGetDeviceCount(&nd));
  printf("Visible GPUs: %d, buffer=%zuMB\n", nd, MB);

  std::vector<uint32_t> h(n);
  int iters = 20;
  // correctness + bandwidth matrix
  for(int i=0;i<nd;i++){
    for(int j=0;j<nd;j++){
      if(i==j){ printf("   self  "); continue; }
      int can=0; CK(cudaDeviceCanAccessPeer(&can,i,j));
      if(!can){ printf("  noP2P  "); continue; }
      CK(cudaSetDevice(i));
      cudaError_t pe = cudaDeviceEnablePeerAccess(j,0);
      if(pe!=cudaSuccess && pe!=cudaErrorPeerAccessAlreadyEnabled){ printf(" enFAIL  "); cudaGetLastError(); continue; }
      uint32_t *src,*dst;
      CK(cudaSetDevice(i)); CK(cudaMalloc(&src,bytes));
      CK(cudaSetDevice(j)); CK(cudaMalloc(&dst,bytes));
      // fill src on GPU i
      CK(cudaSetDevice(i));
      fill<<<(n+255)/256,256>>>(src,n,(uint32_t)(i*131+j));
      CK(cudaDeviceSynchronize());
      // peer copy i->j
      CK(cudaMemcpyPeer(dst,j,src,i,bytes));
      CK(cudaSetDevice(j)); CK(cudaDeviceSynchronize());
      // verify
      CK(cudaMemcpy(h.data(),dst,bytes,cudaMemcpyDeviceToHost));
      size_t bad=0; uint32_t seed=(uint32_t)(i*131+j);
      for(size_t k=0;k<n;k++){ if(h[k]!=(uint32_t)(k*2654435761u+seed)){ bad++; if(bad<=3) printf("[mismatch@%zu got %08x exp %08x]",k,h[k],(uint32_t)(k*2654435761u+seed)); } }
      // bandwidth i->j  (set device i BEFORE creating events; events are bound
      // to the device active at creation, so create+record must share device i)
      CK(cudaSetDevice(i));
      cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      CK(cudaEventRecord(a));
      for(int t=0;t<iters;t++) CK(cudaMemcpyPeer(dst,j,src,i,bytes));
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
      float ms=0; CK(cudaEventElapsedTime(&ms,a,b));
      double gbps = (double)bytes*iters/(ms/1e3)/1e9;
      if(bad) printf(" FAIL%-3zu ", bad>999?999:bad); else printf(" %6.1f ", gbps);
      cudaEventDestroy(a); cudaEventDestroy(b);
      cudaFree(src); cudaFree(dst);
    }
    printf("  <- GPU%d as src (GB/s, row=src col=dst)\n", i);
  }
  printf("done\n");
  return 0;
}
