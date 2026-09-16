# 方案三 review 记录：v1 范围、已修问题与后续注意项

> 日期：2026-06-30  
> 来源：Codex 对当前 `METHOD3` 动态 BAR1 P2P 原型的静态 review。  
> 目的：把今天发现的问题、已经修掉的 correctness blocker、以及未来实现/上机测试时必须继续注意的
> 边界条件集中记录，避免后续误判。

## 0. v1 支持范围建议

第一版建议明确收窄到：

- **同构 GPU 节点**：同一个节点只放同一类卡，尤其是目标 48GB AD102。暂不支持 48GB dynamic 卡与
  正常 static BAR1 卡混插。
- **NCCL 默认 P2P transport**：目标是 NCCL transport FIFO / direct pointer 的显式 external allocation
  映射路径，不承诺支持所有 CUDA/UVM P2P 用法。
- **一进程一 GPU rank 优先**：避免旧式同进程多卡 `cudaDeviceEnablePeerAccess` 打开过大范围的 peer
  访问。
- **关闭大 user-buffer registration**：测试时建议固定
  `NCCL_LOCAL_REGISTER=0 NCCL_GRAPH_REGISTER=0`，避免把 GB 级用户 buffer 映进 BAR1。
- **关闭 cuMem host 路径作为环境规避**：在 L40 / AD102 + 新 CUDA runtime 环境，若 NCCL 初始化会因
  cuMem host allocation 路径崩溃，测试时固定 `NCCL_CUMEM_HOST_ENABLE=0`。
- **不支持 UVM managed peer migration**：`cudaMallocManaged` 跨卡迁移、`uvm_peer_copy=virt`、以及依赖
  UVM 内部物理 peer copy 的路径都不在 v1 范围内。

## 1. 今天确认并修掉的主要 correctness blocker

### 1.1 BAR1 VA 不能假设连续

原始原型调用 `kbusMapFbAperture_HAL(..., BUS_MAP_FB_FLAGS_ALLOW_DISCONTIG, ...)`，但后续只使用
`memArea.pRanges[0]` 并线性编码：

```c
peer_addr = dynBar1DmaBase + offset + i * mappingPageSize;
```

这在 `memArea.numRanges > 1` 时会把第二段之后的地址编码到 BAR1 gap 或错误 window，属于静默数据损坏。

当前修法：

- 去掉 `BUS_MAP_FB_FLAGS_ALLOW_DISCONTIG`，要求 BAR1 VA 单段连续；
- 显式检查 `memArea.numRanges == 1`，否则报错并回滚；
- v1 接受 BAR1 VA 碎片导致 map 失败，而不是回退到多段线性错误编码。

后续如果要支持多段 BAR1 VA，需要像 third-party P2P 那样按 `MemoryArea` range 分段填地址，不能继续只用
一个 `dynBar1DmaBase`。

### 1.2 dynamic window size 不能用 `ActualSize`

第一轮曾考虑用：

```c
RM_ALIGN_UP(pMemDesc->ActualSize, pageSize)
```

作为 dynamic BAR1 window 大小，用来匹配 builder 里 `allocSize` 的校验。但进一步核实后，这个修法是错的：
`kbusMapFbAperture` 最终会走 `memdescCreateSubMem()`，而它要求：

```c
Offset + Size <= pMemDesc->Size
```

也就是不能超过 `memdescGetSize()`。如果 `ActualSize > Size`，用 aligned `ActualSize` 反而会直接创建失败。

当前修法：

- dynamic BAR1 window 大小使用 `memdescGetSize(pMemDesc)`；
- dynamic PTE / physAddr 分支增加防御性检查：`offset + size <= memdescGetSize(pMemDesc)`；
- 理由是 UVM 通过 `nvGpuOpsFillGpuMemoryInfo()` 看到的 allocation size 也是 `memdescGetSize()`，正常
  external allocation map 请求不会超过它。

注意：builder 内部的通用 `size == 0` 语义仍会把 `mappingSize` 解释为 aligned `ActualSize`，见后续注意项。
NCCL/UVM external PTE 路径实际传入非零 size，因此 v1 可接受。

### 1.3 subdevice 创建失败路径的 mutex 泄漏

`pDynBar1Mutex` 创建成功后，如果后续 `pRmApi->Alloc()` 或 `trackDescriptor()` 失败，旧代码直接
`portMemFree(rmSubDevice)`，没有销毁 mutex。

当前修法：

- 在 `cleanup_subdevice_desc` 中，如果 `pDynBar1Mutex != NULL`，先 `portSyncMutexDestroy()` 再 free。

## 2. v1 可接受但必须记住的限制

### 2.1 `size == 0` 的通用接口语义

`nvGpuOpsBuildExternalAllocPtes()` / `...PhysAddrs()` 里：

```c
mappingSize = size ? size : allocSize;
```

而 dynamic 分支新增的越界校验使用的是原始 `size`。如果未来有调用者传 `size == 0` 表示“映射整个
allocation”，当前校验不会拦住 `allocSize > memdescGetSize()` 的尾部。

v1 判断：

- NCCL/UVM external PTE 路径会传入明确的非零 size；
- 因此 v1 可接受。

未来硬化：

- dynamic 分支应改用 `mappingSize` 做越界校验；
- 或者在 dynamic 模式下显式拒绝 `size == 0`；
- 或者把 builder 的 dynamic 模式上限统一改成 `memdescGetSize()`。

### 2.2 BAR1 预算计数不是 remote 全局准入

当前 `dynBar1MappedBytes` 挂在 **source subdevice** 上，但实际消耗的是 **remote GPU 的 BAR1 VA**。
在 8 卡场景里，同一个 remote GPU 会被多个 source GPU 映射；每个 source 只看到自己的计数，无法得到
remote BAR1 的全局占用。

v1 判断：

- NCCL 默认 transport buffer 很小，通常远小于 32GB BAR1；
- 当前预算 guard 只是“早期清晰报错”的 heuristic；
- 真正耗尽仍由 `kbusMapFbAperture` 失败兜底。

未来硬化：

- 预算应挂在 remote GPU / remote `KernelBus` 侧；
- 需要全局 refcount 和 byte accounting；
- 最好配合明确的 admission policy：超预算时返回可识别错误，并确认 NCCL 能优雅回退。

### 2.3 `dynBar1DmaBase != 0` 被用作 dynamic/static 模式标志

当前 builder 用：

```c
if (dynBar1DmaBase != 0) dynamic; else static;
```

理论上 IOVA 0 可以是合法地址。一旦 dynamic window 的 IOVA base 真是 0，会误走 static 分支。

v1 判断：

- 实际平台上分到 0 的概率很低；
- 即便发生，在 static BAR1 未启用时通常会因为 `kbusGetBar1P2PDmaInfo` 失败而干净失败，不太会静默损坏；
- v1 可接受。

未来硬化：

- 增加独立 `NvBool bDynBar1Mapped` 参数，和 `dynBar1DmaBase` 一起传入 PTE / physAddr builder；
- 不再用地址值本身表达状态。

### 2.4 UVM managed memory / peer copy 不在 v1 范围

`nvGpuOpsGetP2PCaps()` 在 dynamic 模式下给 UVM 上报 `bar1DmaAddress/Size = 0/0`，但仍报告
`p2pLink = PCIE_BAR1`。这对 external allocation PTE 路径没问题，因为地址由 RM builder 直接编码；但
UVM 内部 managed memory peer copy 路径会用 `bar1_p2p_dma_base_address` 做物理地址换算。

v1 判断：

- NCCL 显式 buffer 不走 managed memory peer migration；
- v1 文档明确不支持 `cudaMallocManaged` 跨卡迁移和 `uvm_peer_copy=virt`。

未来硬化：

- 对 managed peer copy 做显式 gate，避免地址错误；
- 或在 UVM/RM 接口中区分“BAR1 link exists for external mappings”和“global BAR1 DMA window exists”；
- 如果要支持 managed，需要完全不同的 per-allocation 或迁移路径设计。

### 2.5 同构限定规避了 static/dynamic 混插问题

如果未来支持异构节点，会遇到一类方向性问题：

- A 卡 static BAR1 启用，B 卡 dynamic；
- B 访问 A 的方向仍可能走 static identity；
- 但当前 pair-level static IOMMU 创建逻辑只在两端都 static 时才建，混插时会漏掉单向 static IOMMU。

v1 判断：

- 同构节点下不会出现一端 static、一端 dynamic；
- v1 可不处理。

未来硬化：

- static IOMMU 建拆应按方向判断：remote static 就为该方向建，remote dynamic 就跳过；
- 不能只用“两端都 static”作为 pair-level 条件。

## 3. NCCL / CUDA 环境相关注意项

### 3.1 `NCCL_CUMEM_HOST_ENABLE=0` 是测试规避，不是核心方案依赖

L40 / AD102 在 CUDA runtime > 12.6 时，NCCL cuMem host allocation 路径可能在初始化阶段崩溃或返回异常。
这条路径与本方案的 GPU peer BAR1 PTE 编码不是同一件事，但会挡住 NCCL 初始化，让 dynamic BAR1 P2P
根本跑不到。

建议 v1 测试固定：

```bash
export NCCL_CUMEM_HOST_ENABLE=0
```

注意不要混淆：

- `NCCL_CUMEM_HOST_ENABLE=0`：关闭 NCCL 的 cuMem host shared-memory 路径；
- `NCCL_CUMEM_ENABLE=0`：关闭 NCCL 的 cuMem device allocation 路径。

后者建议做 A/B 测试，不要一开始就固定关闭。

### 3.2 推荐 v1 测试环境变量

```bash
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_LOCAL_REGISTER=0
export NCCL_GRAPH_REGISTER=0
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM
```

关键是确认日志里走的是 P2P/direct pointer，而不是 SHM fallback。

### 3.3 同进程多卡和 legacy peer access

同进程多 GPU、旧式 `cudaDeviceEnablePeerAccess`、或某些 legacy NCCL 路径可能尝试打开更大范围的 peer
访问，甚至接近整设备映射。v1 最好限定：

- 一进程一 GPU rank；
- 默认 cuMem device 路径先保留；
- 如需兼容 `NCCL_CUMEM_ENABLE=0`，要单独确认映射规模仍是 NCCL transport buffer，而不是用户大 buffer。

## 4. 上机测试必须观察的点

### 4.1 功能路径

- `nvidia-smi topo -p2p r/w`：确认 P2P caps 不再因 static BAR1 关闭而失败；
- `p2pBandwidthLatencyTest` / `simpleP2P`：确认 peer 读写正确；
- `nccl-tests all_reduce_perf -g 2`：确认不卡 init，且带宽不是 SHM fallback；
- `NCCL_DEBUG=INFO`：确认 `P2P/direct pointer`。

### 4.2 生命周期 / 泄漏

反复执行：

- NCCL init/destroy；
- 多轮 `all_reduce_perf`；
- 异常退出 / kill rank；
- 重建 CUDA context。

观察：

- dynamic BAR1 map/unmap 日志是否配对；
- `dynBar1MappedBytes` 是否回到 0；
- BAR1 VA 是否随轮次泄漏；
- IOMMU map/unmap refcount 是否平衡；
- GPU 是否出现 P2P 使用后 100% utilization 卡住。

### 4.3 BAR1 碎片

因为 v1 要求单段 BAR1 VA，长期运行后 BAR1 VA 碎片可能导致 `kbusMapFbAperture` 找不到连续区而失败。
这是 v1 有意选择的安全失败模式。

未来如果这成为实际问题，再考虑：

- 实现多段 `MemoryArea` 编码；
- 或加更强的 reuse/refcount；
- 或给 NCCL small buffers 做更稳定的预留/池化。

### 4.4 平台设置

- 关闭或规避 PCIe ACS；
- 确认 IOMMU/ATS/PCIe atomics 行为；
- 记录 BAR1 size、FB size、CUDA driver/runtime、NCCL 版本；
- 失败时优先看 `METHOD3` 的 `LEVEL_ERROR` 日志。

## 5. 后续代码硬化清单

按优先级：

1. **引入 `bDynBar1Mapped`**，消除 `dynBar1DmaBase != 0` 歧义。
2. **dynamic 分支统一用 `mappingSize` 做越界校验**，处理 `size == 0` 泛化语义。
3. **remote-side BAR1 全局预算计数**，替代当前 source-side heuristic。
4. **明确 managed/UVM peer-copy gate**，避免 dynamic BAR1 caps 被误用于 global BAR1 DMA window。
5. **异构 static/dynamic 方向性 IOMMU 建拆**，为未来混插卡支持做准备。
6. **多段 BAR1 VA 支持**，如果实际运行中单段 VA 失败率不可接受。
7. **更完整的失败回退语义**，确认 NCCL user-buffer registration 失败时能回退而不是直接终止。

## 6. 当前结论

在“同构 48GB AD102 + NCCL 默认 P2P + registration 关闭 + managed 不支持”的 v1 范围内，今天发现的两个
主要数据正确性 blocker 已经有合理修复：

- 不再把 discontig BAR1 VA 当连续区；
- 不再尝试超过 `memdescGetSize()` 的 dynamic window。

剩余问题主要是泛化语义、预算精度、生命周期实测和不支持路径的显式 gate。它们不应阻止 v1 上机验证，
但必须在测试文档和后续 hardening 计划中持续跟踪。
