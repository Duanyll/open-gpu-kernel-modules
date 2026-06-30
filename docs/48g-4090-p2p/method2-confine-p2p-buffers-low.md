# 方案二：只把"会被 P2P 共享"的内存钉在低 32GB（保住全部 48GB）

> 目标：让 PMA 照常管理全部 48GB，但**仅把会被 peer 访问的那部分内存**约束在底部 ~31GB，
> static BAR1 也只 identity-map 底部 ~31GB。这样本机数据可用满 48GB，P2P 只占用 BAR1 能覆盖的
> 一小块。**结论：在"纯驱动、不改用户态"的约束下此方案走不通**——卡在一个硬障碍上；但调查过程
> 牵出了更优的方案三（动态 BAR1 P2P），见文末。

## 1. 思路

关键洞察：**本机访问自己显存不经过 BAR1，只有 P2P 才需要 BAR1**。所以理论上：

- PMA 管全部 48GB，普通分配（模型权重等）爱落哪落哪；
- static BAR1 只覆盖底部 ~31GB；
- 把"会被 peer 映射"的分配强制放进底部 ~31GB；
- 对偏移 ≥31GB 的 peer 映射做拒绝/兜底。

可行性取决于两件事：(A) 驱动能不能把 static BAR1 与 PMA 解耦、并把特定分配钉到低地址；
(B) NCCL 到底把什么暴露给 peer，量有多大。

## 2. 调查结论

### 2.1 驱动侧：机制基本都到位（好消息）

- **peer 映射是 per-allocation 的**。走 UVM 的 `UVM_MAP_EXTERNAL_ALLOCATION`，每个内存 handle 一次
  （`kernel-open/nvidia-uvm/uvm_map_external.c:1086`），逐 GPU 调
  `nvUvmInterfaceGetExternalAllocPtes` → `nvGpuOpsGetExternalAllocPtesOrPhysAddrs`
  （`src/nvidia/src/kernel/rmapi/nv_gpu_ops.c:4572`）。每次只处理一个 memdesc，所以**映射时就知道这块
  内存的物理 FB 偏移**。GPU 对之间的 static BAR1 IOMMU 窗口只建一次、引用计数
  （`kbusCreateP2PMappingForBar1P2P_GH100`，`kern_bus_gh100.c:1704`），不枚举具体分配。

- **static BAR1 区就是 P2P 窗口**。`kbusGetBar1P2PDmaInfo_GH100`
  （`kern_bus_gh100.c:1657-1664`）返回的 DMA 窗口大小 = `memdescGetSize(staticBar1.pDmaMemDesc)`，
  IOMMU 也是整段映射 `staticBar1.pDmaMemDesc`。**所以把 static BAR1 缩到底部 ~31GB，P2P 窗口自动
  跟着缩到底部 ~31GB**，正合我们意图。

- **驱动本来就能处理"分配落在 static 区之外"**。`kbusGetStaticFbAperture_TU102`
  （`kern_bus_tu102.c:1091-1135`）把每个请求范围分类：limit 超过 `staticBar1.size` 的走
  **dynamic BAR1 回退**（返回 `NV_ERR_NOT_SUPPORTED`，上层改用普通动态映射）。所以
  **PMA 跨全 48GB、static BAR1 只覆盖底部**这件事，架构上是支持的——高地址分配照样能本机用，只是
  进不了 P2P identity 快路径。需要改的就两处：`kbusIsStaticBar1Supported_TU102` 的尺寸检查
  （`:432`/`:498`）和 `kbusEnableStaticBar1Mapping_TU102` 的映射尺寸（`:551`），把 client FB 解耦成
  `min(clientFB, bar1 - headroom)`；外加 RUSD 统计微调（`mem_mgr.c:3238`）。512MB 对齐需保持
  （`kern_bus_gh100.c:1565`）。

- **把分配钉到低地址的原语存在**。PMA 支持 `PMA_ALLOCATE_SPECIFY_ADDRESS_RANGE` +
  `physBegin/physEnd`（`phys_mem_allocator.h:88-118`，强制到位图扫描 `regmap.c:1480-1626`）；
  heap 支持 `NVOS32_ALLOC_FLAGS_USE_BEGIN_END` + `rangeLo/rangeHi`（`nvos.h:1620-1621`，
  `heap.c:1421-1704`）。`_vidmemPmaAllocate` 会把 `rangeLo/rangeHi` 直接翻成
  `physBegin/physEnd`（`video_mem.c:248-321`）。

### 2.2 NCCL 侧：被 peer 看见的内存确实小且由 NCCL 自己分配（好消息）

基于 NCCL `2.30.7` 源码（`/tmp/nccl_src`，已 clone）：

- **默认（不开 user-buffer 注册）下，PCIe P2P 只把 NCCL 自己的传输 FIFO 暴露给 peer**，不是用户
  buffer。peer 端拿到的就是 `ncclSendMem`/`ncclRecvMem` + 每协议数据 buffer
  （`src/transport/p2p.cc:407-410,486-490`；`src/include/comm.h:53-78`），由
  `ncclP2pAllocateShareableBuffer`（`ncclCudaCalloc`/`cuMemAlloc`，`p2p.cc:220-258`）分配。
- **数据路径**：本机 kernel 把用户 buffer 拷进 FIFO，邻居 GPU 通过 P2P 读 FIFO
  （`src/device/prims_simple.h:496,538-547`，peer 指针指向 `conn->buffs[...]` 即 NCCL FIFO，
  不是用户 buffer）。LL/LL128 永远只用 FIFO。
- **量级**：`NCCL_BUFFSIZE` 默认 4MiB，每连接 ~6–8MiB，整卡量级 **几十 MB ~ ~100MB**，与 48GB 用户
  数据无关。
- **用户 buffer 被 peer 映射只在显式 opt-in 时发生**：`ncclCommRegister` 或 CUDA graph 捕获
  （`NCCL_LOCAL_REGISTER`/`NCCL_GRAPH_REGISTER`，默认 1 但仅在有注册/图捕获时才触发，
  `src/register/coll_reg.cc:339-383`），且只映射被注册的那一块。可用
  `NCCL_LOCAL_REGISTER=0 NCCL_GRAPH_REGISTER=0` 关掉。
- **单进程多卡 + 旧式 `cudaDeviceEnablePeerAccess`** 会在驱动层打开整块 peer 设备
  （`p2p.cc:351`）；但 cuMem 路径（`NCCL_CUMEM_ENABLE`，新驱动默认）即使同进程也只按 handle 映射。
  生产常态是一进程一卡（多进程，逐 buffer IPC），不受此影响。

### 2.3 决定性障碍：分配时**没有**"可导出/可共享"信号（坏消息）

要"只把会被 P2P 的分配钉到低地址"，前提是分配那一刻能识别它是不是 peer-destined。
**RM 在物理分配路径上做不到这一点：**

- 可共享的 `cuMemCreate`（带 shareable handle）与普通 `cudaMalloc`，**走的是完全相同的**
  `vidmemConstruct_IMPL` → `_vidmemPmaAllocate` → `pmaAllocatePages`，传入的
  `NV_MEMORY_ALLOCATION_PARAMS` 也完全相同（`video_mem.c:641`；`nvos.h:1603-1637` 里**没有**
  export/ipc/shareable/fabric 字段）。
- "可导出"是**事后**叠加在已落地物理内存上的独立对象：`NV_MEMORY_EXPORT`(cl00e0) 明确
  "No memory is allocated"（`cl00e0.h:38`），靠 `DupObject` 包已有 handle（`mem_export.c:643`）；
  `NV_MEMORY_FABRIC`(cl00f8) takes `map.hVidMem` = 已分配 handle（`cl00f8.h:78-86`，
  `mem_fabric.c:539`）。`memIsExportAllowed` 对所有 Memory 无条件返回 `NV_TRUE`
  （`g_mem_nvoc.h:630-632`）——导出门槛在**导出时**判，不在分配时。
- 旧式 `cudaMalloc`+`cudaIpcGetMemHandle` 同理：任何 LOCAL_USER 显存都能 IPC，无分配时标记。

也就是说：**range 约束的水管（`physBegin/physEnd`、`rangeLo/rangeHi`）齐全且生效，但分配那一刻没有
任何可靠信号告诉驱动"这块以后会被 P2P 共享"**，所以无法只对这类分配施加低地址约束。
注入点很明确（`video_mem.c:248`/`:321`），缺的是触发条件。

## 3. 在"纯驱动、不改用户态"约束下的判定

要补上那个信号，只能：(a) 改闭源 CUDA UMD（不可能）；(b) 让内核驱动在分配时看到 export 意图
（不存在）；(c) 用全局/启发式策略（如"所有分配都靠下"——那等于方案一，没收益；或按大小启发式——
太脏）。本项目的核心约束恰恰是**不改用户态、纯靠 patch 驱动**，而：

- NCCL 侧配合（改 NCCL 源码 / 设环境变量去钉低地址）违背"无需改用户态"的初衷，且公开 CUDA API
  **根本不暴露物理 FB 放置**，NCCL 自己也没法请求"分配到低地址"；
- 驱动侧没有可用的 per-allocation 信号。

**所以方案二（精准只钉 P2P 内存）在纯驱动约束下不成立。** 唯一不依赖信号的退路是"不钉、只在
peer-map 超界时拒绝 + 指望 NCCL 优雅回退到 FIFO 慢路径"，但这既脆弱（取决于分配顺序与 PMA 放置
方向）又把带宽打回慢路径，得不偿失。

## 4. 调查牵出的更优方向 → 方案三：动态 BAR1 P2P

把问题反过来看就豁然开朗：**既然被 peer 共享的集合很小（几十 MB）、而且是被显式逐个映射的（不是
缺页触发的）**，那就根本不需要把整块 FB identity-map 进 BAR1。可以**在每次 peer-map 某个分配时，
用现成的动态 BAR1 机制（`kbusMapFbAperture`）把这一个分配映射进 target GPU 的 BAR1，拿到它的 BAR1
总线地址，再让 source 的 peer PTE 指向 `targetBar1Base + 该分配的动态 BAR1 偏移`**。

- 这恰好绕开了"透明分页做不到"的死结：之前说不可行，是因为缺页驱动的换页在 inbound BAR1 上不可
  重放；而这里 **CUDA/NCCL 已经显式告诉驱动要映射哪些分配**，驱动**预先 eager 映射**即可，无需
  缺页重试。
- BAR1 预算：只要被 peer 映射的总量 ≤ 32GB 即可。NCCL 默认 P2P 共享集只有几十 MB，**绰绰有余，
  且保住全部 48GB**。开了 user-buffer 注册的大 buffer 若超预算，可拒绝该 peer-map → NCCL 回退到
  FIFO（待验证其优雅性）。
- 带宽：动态 BAR1 映射就是真实的 GMMU→BAR1 映射，访问是满 PCIe 带宽，与 static identity 无异。
- 代价：要在 `nvGpuOpsBuildExternalAllocPtes` / `kbusGetBar1P2PDmaInfo` 这条链上，把"用 FB 偏移
  identity 编码"改成"为每个 peer-mapped 分配建动态 BAR1 映射并用其 BAR1 偏移编码"，外加生命周期/
  引用计数/IOMMU 逐块映射的管理。是实打实的驱动开发，但都基于已有原语，且**纯驱动、不改用户态**。

这才是真正能"保住 48GB + 全带宽 P2P"的路子，下一篇 `method3-*.md` 专门论证它。

## 5. 结论

- 方案二的**机制拼图基本齐全**（per-alloc 映射、static BAR1 可缩、dynamic 回退已存在、range 约束
  原语可用），NCCL 的 P2P 共享集也确实小且自管理。
- 但**唯一缺的那块——分配时识别"可共享"——在纯驱动约束下补不上**，导致"精准只钉 P2P 内存"不可行。
- 真正的出路是**方案三：动态 BAR1 P2P**——利用"共享集小且被显式映射"这一事实，逐分配动态映射进
  BAR1，无需 identity-map 全 FB，保住 48GB。建议下一步深入方案三。

## 附：本次调查的代码锚点
- 驱动 peer-map 路径 / 无 export 信号：`nv_gpu_ops.c:4572,4708-4727,4182-4212`、
  `video_mem.c:248-321,641`、`mem_export.c:566,643`、`mem_fabric.c:539`、`g_mem_nvoc.h:630-632`、
  `nvos.h:1603-1637`、`cl00e0.h:38`、`cl00f8.h:78-86`
- static BAR1 解耦 / range 约束：`kern_bus_tu102.c:432,498,551,1091-1135`、`kern_bus_gh100.c:1657-1664,1565`、
  `phys_mem_allocator.h:88-118`、`phys_mem_allocator.c:654,852-855`、`heap.c:1421-1704`
- NCCL：`/tmp/nccl_src/src/transport/p2p.cc:220-258,407-410,486-490,345-386`、
  `src/include/comm.h:53-78`、`src/device/prims_simple.h:496,538-547`、`src/register/coll_reg.cc:339-383`、
  `src/init.cc:810-827`
