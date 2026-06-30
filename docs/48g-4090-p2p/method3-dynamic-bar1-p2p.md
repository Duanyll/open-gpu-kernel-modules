# 方案三：动态 BAR1 P2P（逐分配映射，保住全部 48GB）

> 核心：不再像 GH100 那样把整块 FB identity-map 成"一条大映射条目"，而是**每次 peer-map 一个分配
> 时，用现成的动态 BAR1 机制把这一块映射进 target 的 BAR1**，让 source 的 peer PTE 指向它的 BAR1
> 总线地址。被 peer 共享的集合很小且是显式映射的，所以 32GB BAR1 装得下，**48GB 全部可用、满带宽、
> 纯驱动不改用户态**。这是唯一同时满足三项约束的方向。

## 1. 为什么这条路成立（而"透明分页"不成立）

之前判死"透明分页 BAR1 P2P"，是因为 **缺页驱动**的换页在 inbound BAR1 上不可重放。但方案三**不靠
缺页**：CUDA/NCCL 通过 `UvmMapExternalAllocation`（per-allocation）**显式告诉驱动要映射哪些分配**，
驱动 **eager 预映射**即可。把"按需缺页"换成"按显式请求预映射"，死结就解开了。

## 2. 你的关键问题的答案：AD102 能放多少条映射条目？够 NCCL 用吗？

**够，且富余 2–3 个数量级。**

### 2.1 容量侧：没有"槽位数"上限，只受 32GB BAR1 VA 约束

- BAR1 VA 空间大小 = 物理 BAR1 大小：`apertureLength = kbusGetPciBarSize(pKernelBus,1)`
  （`kern_bus.c:567`），32GB BAR1 → 32GB VA（`kern_bus_gm107.c:900,1031`）。
- **没有任何固定数组 / `MAX_*` 限制 BAR1 映射条数**。所有 per-mapping 簿记都是动态分配的
  （eheap 按需分配块 `eheap_old.c:101`；reuse DB 红黑树 `mapping_reuse.h`；引用计数）。
- BAR1 页表在 **FB 中按需懒分配**（PDE 级 sparse，PTE 表用到才建，`mmu_walk.c:420-428,1514-1521`），
  **不是**预留的固定池（BAR1 不走 `PTETABLE_PMA_MANAGED`，`gpu_vaspace.c:1057-1065`）。
- 默认页大小：32GB BAR1 不强制 64KB（`kbusDetermineBar1Force64KBMapping`：>256MB 时
  `bBar1Force64KBMapping=NV_FALSE`，`kern_bus.c:537-540`），FB 分配按其原生 2MB 大页映射 → PTE 很省。
- 可映射非连续分配（走 memdesc 页表，`virt_mem_allocator_gm107.c:794-798,1165-1172`）。

**结论**：实际可支撑**数千到上万**条几 MB 的独立映射才会触及 VA 压力，远超 NCCL 所需。

### 2.2 决定性证据：GDR / 第三方 P2P 已经在干这件事（现成先例）

驱动里**已经有**一条把**单个 VRAM 分配按需映射进 BAR1、并交出 PCIe 总线地址**的成熟通路——
**GPUDirect RDMA / 第三方 P2P**：

```
nvidia_p2p_get_pages (nv-p2p.c:650)            ← nvidia-peermem 调用
  → RmP2PGetPages (p2p.c:1505)
    → RmThirdPartyP2PBAR1GetPages (p2p.c:489)
      → _createThirdPartyP2PMappingExtent → kbusMapFbAperture_HAL (p2p.c:212)   ← 逐分配 BAR1 映射
      → 总线地址 = BAR1 base + BAR1 VA (p2p.c:466)
```
注释原话（`p2p.c:479-487`）："creates mappings from BAR1 VASpace for registered third party P2P
allocations … BAR1 addresses, BAR1 base + BAR1 VAs returned by RM."

- **准入控制就是字节预算**，不是条数：`_isSpaceAvailableForBar1P2PMapping`
  （`p2p.c:994,1036`）只比较累计 `P2PfbMappedBytes` 与 `bar1Size − 36MB`
  （`CLI_THIRD_PARTY_P2P_BAR1_RESERVE`，`g_third_party_p2p_nvoc.h:89`）。条数无上限。
- **逐分配跟踪 + 引用计数 + 拆除**齐全（`CLI_THIRD_PARTY_P2P_MAPPING_EXTENT_INFO`，建于
  `_createThirdPartyP2PMappingExtent`，拆于 `RmThirdPartyP2PMappingFree` → `kbusUnmapFbAperture`，
  `p2p.c:103-125,306,384`）——**这正是方案三要复用的模板**。
- **不限数据中心卡**：唯一 construct 门槛是 `PDB_PROP_GPU_COHERENT_CPU_MAPPING`（C2C 相干系统如
  Grace-Hopper 走 NVLINK 类型），消费 PCIe 卡走 BAR1 分支（`third_party_p2p.c:114-122`）。
- `P2P_MAX_NUM_PEERS=8` 只是**旧 mailbox**的限制，与 BAR1 映射条数无关。

> 也就是说：**"逐分配动态映射进 BAR1"在 NVIDIA 自己的驱动里已经规模化运行多年**，只是消费者是 NIC
> 而非 peer GPU。方案三 = 把同一套 `kbusMapFbAperture` 映射，改造成给 peer GPU 用。

**重要修正（来自实测资料）**：曾以为"消费卡加载 peermem 即可跑 GDR"，但据 GitHub issue
[aikitoria #20] 与博客 [harrychen.xyz, 2026-05-20] 的实测，**消费卡（5090, 驱动 590.48.01）的 GDR
开箱不通，且真正的门禁在用户态 `libcuda.so`，不在内核**：

- 作者强行在内核把 `dma_buf_supported=YES`（`nv-dmabuf.c` 等）**无效**——内核早已报 YES，但 CUDA
  仍报 0。真正的 gate 是 libcuda 里 `cuDeviceGetAttribute` 对属性 116 的检查
  `dev_struct[0x5f14] & 0x20 && global_gdr_enabled`。一个**单字节 patch**（`0x42d69b: 0x40→0x60`）
  翻转该位，即点亮 `DMA_BUF_SUPPORTED` / `GPU_DIRECT_RDMA_SUPPORTED` / `..._WITH_CUDA_VMM`。
- 用的是 **dma-buf** 路径（`cuMemGetHandleForAddressRange` / `perftest --use_cuda_dmabuf`），**不是**
  老的 `nvidia-peermem`。打完 patch 后端到端可用：NCCL 24 卡 3 机 400G IB，busbw **8.87→19.93 GB/s**，
  日志出现 `via NET/IB/0/GDRDMA`。作者结论：**硬件没被砍，纯软件 gate**（同硅的 RTX Pro 6000 支持 GDR）。

这对方案三是**双重利好 + 一条警示**：
- *利好 1*：证明**内核侧把 GPU 显存映进 BAR1 给外部 DMA 的机制，在消费 Blackwell 上物理可用**
  （libcuda 解禁后 GDR 端到端跑通）——即 BAR1 逐分配映射的底层在消费硅上是活的。
- *利好 2*：现代 GDR 走 **dma-buf（`nv-dmabuf.c`）**，这是除 `third_party_p2p.c` 之外**第二条**
  "逐分配导出 GPU 内存的 BAR1 总线地址"的现成先例。
- *警示*：能力可能**同时被内核与 libcuda 两层 gate**。但方案三是**节点内 GPU↔GPU P2P**，其能力位
  （`cudaDeviceCanAccessPeer` / `NV0000_CTRL_SYSTEM_GET_P2P_CAPS` 的 `isBar1P2PSupported`）来自**内核**，
  且**本仓库现有 mod 已经在不打任何 libcuda patch 的情况下让 NCCL P2P 端到端跑通**——所以这层 gate
  对节点内 P2P 是经验上已解决的，方案三仍是**纯内核、零用户态改动**。这点与需要 libcuda patch 的
  **跨节点 GDR** 不同，务必区分。

### 2.3 需求侧：NCCL 实际需要多少条（NCCL 2.30.7 源码核对）

规则：**每个 (peer, channel, direction, connIndex) 连接一个独立 cuMem/IPC 分配，无跨 channel 合池**；
3 个协议（LL/LL128/SIMPLE）共用同一分配（不翻倍），每个约 **6 MiB**。

| 拓扑（PCIe，无 NVLink，registration 关） | 每 GPU 暴露的独立映射数 | 每 GPU 总字节 |
|---|---|---|
| 2-GPU all-reduce(ring) | ~4 | ~24 MB |
| 8-GPU all-reduce(ring) | ~16 | ~96 MB |
| 8-GPU all-to-all(sendrecv) | (nPeers−1)×p2pChPerPeer ≈ 7×2 = ~14 | ~84 MB |
| **8-GPU 同时跑 coll + a2a（典型默认）** | **~30** | **~180 MB** |
| 结构硬上限（channel 强行拉满 64） | ~(nPeers−1)×64×连 ≈ 低数千 | — |

**对比**：默认 ~30 条 / ~180MB ↔ 容量数千条 / ~32GB。**富余 2–3 个数量级。**

## 3. 实现蓝图（把"静态按对" P2P 改成"动态按分配"）

以 `third_party_p2p.c` 为模板，改 4 处（agent 已定位）：

1. **IOMMU 映射从"整段静态区按 GPU 对一次"改成"按分配的 BAR1 窗口"**。现状
   `_kbusCreateStaticBar1IOMMUMapping` 把整块 `staticBar1.pDmaMemDesc` 一次性
   `memdescMapIommu(..., pSrcGpu->busInfo.iovaspaceId)`（`kern_bus_gh100.c:1552-1557`）。改为对每个
   peer-mapped 分配的 BAR1 子窗口做 `memdescMapIommu` 进 source 的 IOVA。
   *（与 GDR 的差异：GDR 只产生 NIC 用的总线地址，无需映进 source GPU 的 IOVA；P2P 多这一步。）*
2. **`busBar1PeerRefcount[gpuInstance]`（按对）→ 按分配跟踪 + 引用计数**，照搬
   `CLI_THIRD_PARTY_P2P_MAPPING_EXTENT_INFO` 的建/refcount/拆模式。
3. **编码改用"该分配 BAR1 窗口的 DMA 地址"**。`_nvGpuOpsEncodeBar1P2PAddrs`
   （`nv_gpu_ops.c:3846-3857`）现在加的是按对的 `dmaBaseAddress` + 全 FB 偏移；改为加该分配动态
   BAR1 窗口的 IOVA + 窗口内偏移。所需输入（peer `pMemDesc`、peer GPU、`pMappingGpu`、`offset`）
   在调用点（`nv_gpu_ops.c:4185-4216`）已齐备。
4. **放宽 512MB 对齐断言**（`kern_bus_gh100.c:1565`，仅因静态映射整 FB 而存在）→ 改为按分配页大小
   （如 64KB，GDR 路径就这么对齐的）。

不需要 static BAR1，因此 `OverrideFbSize`、`RMForceStaticBar1` 都不用；PMA 照常管全 48GB。

## 4. 风险与缓解

- **user-buffer registration 是唯一能撑爆 BAR1 预算的情形**：`ncclCommRegister` / CUDA graph 捕获
  会把**整个用户基础分配**（可能是 GB 级）按 peer 映射进 BAR1（`p2p.cc:1010-1177`，每 peer 一份）。
  N 个大 buffer × (nPeers−1) 可能 > 32GB。
  - 缓解：复用 GDR 的字节预算准入（`_isSpaceAvailableForBar1P2PMapping` 模式），超预算就**拒绝该
    peer-map**；需确认 NCCL 在注册失败时优雅回退到 FIFO 拷贝路径（注册是优化，默认路径是 FIFO，
    几十 MB，永远装得下）。或直接 `NCCL_LOCAL_REGISTER=0 NCCL_GRAPH_REGISTER=0` 关掉。
- **同进程多卡 + 旧式 `cudaDeviceEnablePeerAccess`** 会在驱动层开放整块 peer 设备；强制 cuMem
  （`NCCL_CUMEM_ENABLE=1`，新驱动默认）或一进程一卡可避免。
- **生命周期/拆除**要跟住 UVM external map 的建立/销毁，避免泄漏 BAR1 VA（GDR 的 refcount 模板已解决
  这类问题）。

## 5. 为什么动态映射能吃下 >32GB 偏移的分配（方案三的真正机理）

这是方案三与现有 static 方案的本质区别，也是它能保住 48GB 的根因：

- **static BAR1**：bus 地址 = `BAR1base + fbPhysOffset`。分配的物理偏移 >32GB 时，bus 地址越过 BAR1
  窗口 → 损坏。所以要求 BAR1 ≥ FB。
- **dynamic BAR1**（`kbusMapFbAperture`，GDR/第三方 P2P 同款）：把该分配映射进一个 **BAR1 VA 槽**
  （偏移落在 0..32GB 内），并用 **GMMU PTE 把这个 BAR1 VA → 该分配的 FB 物理页（在 48GB 里的任意
  位置）**。bus 地址 = `BAR1base + bar1VAoffset`，**永远在 32GB 内**；peer DMA 打到 BAR1 孔径，GMMU
  把它翻译到高地址 FB 页。**所以分配落在 48GB 的任何地方都行，只有"同时映射的 BAR1 VA 总量"需 ≤32GB**
  （NCCL 默认 ~180MB）。GMMU 的这层间接，正是它优于 static 的地方。

## 6. 前置验证（可选，用于经验背书）

两类证据已基本坐实机制，可按需补实测：

1. **节点内 P2P 在消费卡上已端到端跑通（纯内核，无 libcuda patch）**：本仓库现有 static BAR1 mod
   + harrychen 2026-03 文章（5090，`p2pBandwidthLatencyTest` 单向 31.5→48.5 GB/s、双向 32→97.4、
   NCCL 8 卡 busbw 14.75→27.34）。这证明 BAR1 P2P 的 PTE 编码 + IOMMU 在消费硅上 work——只是 5090
   是 32G BAR1=32G VRAM，static 够用；48GB 卡卡在 BAR1<FB，需方案三。
2. **动态 BAR1 映射吃高偏移分配**：可在 48GB 卡上验证——但注意现代 GDR 走 **dma-buf** 且**额外被
   libcuda gate**（见 §2.2），需先打 harrychen 的单字节 libcuda patch，再 `perftest --use_cuda_dmabuf`；
   若把 RDMA buffer 强制分配到 >32GB 物理偏移仍能正确 RDMA，即直接证明 §5 的机理在该卡成立。
   （注：**不要**指望老 `nvidia-peermem` 路径——实测它在改过的驱动里不工作。）

实测 P2P 时的实践注意（来自 harrychen 2026-03）：关闭 PCIe **ACS**；`NCCL_P2P_LEVEL=SYS` 并用
`NCCL_DEBUG=INFO` 确认 `via P2P/direct pointer`；留意 P2P 用后某些卡停在 100% 利用率的资源泄漏
（建/销一个 CUDA context 可清除，可用 systemd timer 自动化）。

## 7. 结论

- **容量问题答案明确**：AD102 的 BAR1 映射**无固定条数上限**，只受 32GB VA 约束，可容纳数千条；
  NCCL 默认仅需 ~30 条 / ~180MB，富余 2–3 个数量级。
- **机制有现成且已规模化的先例**：GDR / 第三方 P2P（`RmThirdPartyP2PBAR1GetPages` →
  `kbusMapFbAperture`）就是逐分配 BAR1 映射，且在消费 Ada 上可用。
- **工程量真实但有界**：核心是把"按 GPU 对的整段静态映射"改造成"按分配的动态映射 + source IOVA +
  peer PTE 编码"，4 处改动，可照搬 `third_party_p2p.c` 模板。
- **唯一需守住的是 BAR1 字节预算**（针对 registration 开启时的大用户 buffer），用 GDR 同款准入 +
  回退即可。

这就是你最初设想的"正道"的正确、可落地形态。建议下一步：按 §3 蓝图，参照 `third_party_p2p.c` /
`nv-dmabuf.c`，把 `nvGpuOpsBuildExternalAllocPtes` 的静态编码改成动态逐分配映射做原型。

## 外部资料

- harrychen.xyz 2026-03-22《Enable PCIe P2P on RTX 5090》：节点内 P2P **纯内核**方案（~100 行，
  patch vtable 用数据中心实现 + 关检查），5090 BAR1=32G=全显存；含 p2pBandwidthLatencyTest / NCCL
  实测与 ACS/`NCCL_P2P_LEVEL=SYS`/资源泄漏等实践注意。**未涉及 BAR1<FB 与动态映射**——即公开工作止于
  本问题起点，佐证 48GB 解法是新工作。
- harrychen.xyz 2026-05-20《Enable GPUDirect RDMA on RTX 5090》：GDR 真正门禁在 **libcuda**
  （单字节 patch `0x42d69b 0x40→0x60`），走 **dma-buf** 非 peermem；解禁后端到端 GDR 可用。见 §2.2。
- GitHub `aikitoria/open-gpu-kernel-modules` issue #20：消费 5090 GDR 开箱不通（未解答）。

## 附：代码锚点
- GDR 先例：`src/nvidia/src/kernel/gpu/bus/p2p.c:212,306,384,466,479-487,489,994,1036,1505`、
  `src/nvidia/src/kernel/gpu/bus/third_party_p2p.c:103-125,114-122`、`kernel-open/nvidia/nv-p2p.c:650`、
  `kern_bus.c:1299`、`g_third_party_p2p_nvoc.h:82,89`
- 容量：`kern_bus_gm107.c:900,1031,1047-1048,537-540`、`kern_bus.c:567`、
  `gpu_vaspace.c:1057-1065`、`mmu_walk.c:420-428,1514-1521`、`g_kern_bus_nvoc.h:249-270,570`
- 改造点：`kern_bus_gh100.c:1538-1573,1565,1636-1664,1688-1737`、`nv_gpu_ops.c:3846-3857,4185-4216`、
  `kbusMapFbAperture_GM107` `kern_bus_gm107.c:3018`
- NCCL：`/tmp/nccl_src/src/transport/p2p.cc:220-258,486-490,606-616,707,731,1010-1177`、
  `src/transport.cc:25-34,171-185`、`src/graph/paths.cc:957-991`、`src/include/device.h:91,231`、
  `src/init.cc:810-827`
