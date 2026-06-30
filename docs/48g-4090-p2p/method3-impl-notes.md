# 方案三实现笔记（动态 BAR1 P2P）— 进度与待办

> 配套设计见 `method3-dynamic-bar1-p2p.md`。本文件记录代码实现的实际进度、关键决策、**尚未完成
> 的必需改动**，以及编译/测试计划。所有改动用 `METHOD3` 注释标记，便于检索与回退。
> 无法在开发机（macOS）编译，需在目标机上 compile-iterate。

## 已完成：`src/nvidia/src/kernel/rmapi/nv_gpu_ops.c`

把"整 FB static identity 编码"换成"逐分配动态 BAR1 窗口编码"，生命周期挂在 duped peer-memory
句柄上：

1. **跟踪结构 + 链表**：`subDeviceDesc` 加了 `pDynBar1List` + `pDynBar1Mutex`；新结构
   `gpuDynBar1P2PMapping`（key = duped 句柄）。
2. **helper**（定义在 `nvGpuOpsGetExternalAllocAperture` 之前）：
   - `_nvGpuOpsDynBar1Create`：`kbusMapFbAperture_HAL`(remote) → 建 ADDR_SYSMEM 窗口 memdesc
     (`remoteBar1Phys + bar1Offset`) → `memdescMapIommu`(source IOVA) → 取窗口 DMA base。
   - `_nvGpuOpsDynBar1Find` / `_GetOrCreate`（mutex 保护，懒创建）/ `_Destroy` / `_DestroyAll`。
3. **编码**：`nvGpuOpsBuildExternalAllocPtes` 与 `...PhysAddrs` 的 `isBar1P2PSupported` 分支改为
   `physicalAddresses[i] = dynBar1DmaBase + offset + i*mappingPageSize`（线性，窗口内连续）。
   新增形参 `RmPhysAddr dynBar1DmaBase`；orchestrator
   `nvGpuOpsGetExternalAllocPtesOrPhysAddrs` 在 `isBar1P2PSupported` 时调 `_GetOrCreate` 取 base 并
   下传。删除了不再使用的 `_nvGpuOpsEncodeBar1P2PAddrs`。
4. **生命周期**：创建 = 首次 build 懒创建；销毁 = `nvGpuOpsFreeDupedHandle` 调 `_Destroy`；
   subdevice 拆除时 `_DestroyAll` 兜底 + 销毁 mutex；mutex 在 `nvGpuOpsRmSubDeviceCreate` 创建。
5. 锁序：RMAPI → `pDynBar1Mutex` → remote GPU lock（创建时若未持有则获取），一致无环。

## ⚠️ 未完成（必需）：`kern_bus_gh100.c` 的能力门禁解耦

**当前编码改动还不会在 48GB 卡上生效**，因为 BAR1 P2P 能力位仍被 static BAR1 卡住：

- `kbusIsPcieBar1P2PMappingSupported_GH100`（`kern_bus_gh100.c:1428`）在两卡未启用 static BAR1 时
  返回 `NV_FALSE`（:1453-1458）。48GB 卡 BAR1<FB → static BAR1 关 → 此函数 false →
  `p2pCaps.bar1Supported` / `isBar1P2PSupported` 恒 false → **动态编码分支永不执行**。

因此还需要（task #5）：

1. **放开能力门禁**：`kbusIsPcieBar1P2PMappingSupported_GH100` 去掉/放宽
   `!kbusIsStaticBar1Enabled(...)` 要求（动态映射不需要 static BAR1）。建议新增一个判定（例如
   "PCIe 且对端可达且非 loopback"，或用 regkey 显式开启动态 BAR1 P2P），替代 static 检查。
2. **每对建链不再依赖 static 区**：`kbusCreateP2PMappingForBar1P2P_GH100` /
   `kbusRemoveP2PMappingForBar1P2P_GH100`（:1705/:1757）在 static BAR1 关时**跳过**
   `_kbusCreateStaticBar1IOMMUMappingForGpuPair`（per-allocation IOMMU 已在 nv_gpu_ops 里做），
   仅保留 refcount。
3. **修 `nvGpuOpsGetP2PCaps`**（`nv_gpu_ops.c:3282-3310`）：那里两处 `kbusGetBar1P2PDmaInfo_HAL`
   读 `staticBar1.pDmaMemDesc`，无 static 区时返回 `NV_ERR_NOT_SUPPORTED` → caps 查询失败。动态模式下
   没有"单一整区 DMA 地址"，应设 `bar1DmaAddress[i]=0 / bar1DmaSize[i]=0` 并仍报
   `p2pLink = UVM_LINK_TYPE_PCIE_BAR1`。**必须先核实 UVM 如何消费这两个值**（见下）。

## 待核实（影响正确性）

- **UVM 如何使用 caps 的 `bar1DmaAddress`/`bar1DmaSize`**（`kernel-open/nvidia-uvm/uvm_gpu*.c`）：
  是否仅信息性、是否会做区间校验/断言。若会校验非零或用于地址换算，则动态模式需要别的上报方式。
- **`kbusMapFbAperture` 在 orchestrator 锁上下文能否跑**：当前在 RMAPI 读锁下、必要时获取 remote
  GPU 写锁。确认无锁序/上下文问题（third_party_p2p 在持 GPU 锁的上下文调用它）。
- **`memArea.pRanges` 所有权**：假定 `kbusMapFbAperture` 分配、`kbusUnmapFbAperture` 释放（对称，
  同 third_party_p2p）。需确认 unmap 确实释放 pRanges，否则泄漏。
- **预算/回退**：当前无显式 BAR1 字节预算；超额时依赖 `kbusMapFbAperture` 自身失败返回错误（dup/
  map 失败 → NCCL 是否优雅回退到 FIFO？默认 FIFO 集 ~180MB 不会触发）。后续可仿
  `_isSpaceAvailableForBar1P2PMapping`（`p2p.c:994`）加准入 + 优雅拒绝。
- **多源共享同一分配**：每个 (源 GPU, dup) 各建一份窗口 memdesc/IOMMU；remote BAR1 映射经
  `kbusMapFbAperture` reuse DB 引用计数共享。确认 refcount 平衡（每 dup 一次 map、free 一次 unmap）。

## 编译 / 测试计划（目标机）

1. 按 README/DKMS 流程把改过的 `src`、`kernel-open` 覆盖进 nvidia-open dkms，`dkms build/install`，
   `modprobe nvidia`。先解决编译错误（本改动未在编译器下验证过）。
2. 完成 task #5 后：`nvidia-smi topo -p2p r/w` 期望两卡 `OK`（动态 BAR1 P2P 已选中）。
3. CUDA `p2pBandwidthLatencyTest` / `simpleP2P` 通过，带宽为 P2P 级。
4. `nccl-tests` `all_reduce_perf -g 2`（同机）不卡 init；`NCCL_DEBUG=INFO` 见 `via P2P/direct
   pointer`。
5. `dmesg` 关注 `METHOD3` 错误打印（"dynamic BAR1 P2P window not set" 等）。
6. 实践：关 PCIe ACS；`NCCL_P2P_LEVEL=SYS`；留意 P2P 用后 100% 利用率泄漏（建/销 CUDA ctx 清除）。

## 更新：task #5 已完成 + 设计改为自适应

后续把方案改成**自适应**（不回归正常卡）并完成了 kern_bus 门禁解耦：

- **自适应 static/dynamic**：编码按 remote 是否启用 static BAR1 二选一——
  - static BAR1 开（正常 4090/5090，BAR1≥FB）→ 走原 **static identity** 路径
    （`kbusGetBar1P2PDmaInfo` + `dmaBase + fbOffset`），**行为不变**；
  - static BAR1 关（48GB 4090，BAR1<FB）→ 走 **per-allocation 动态窗口**。
  判定用 `kbusIsStaticBar1Enabled(remote)`（orchestrator 据此决定是否 `_GetOrCreate` 动态窗口；
  `dynBar1DmaBase != 0` 即动态分支）。**这点很关键**：正常卡 static 区已占满 BAR1 VA，若无条件走动态会
  因 VA 耗尽而失败/回归。
- `kbusIsPcieBar1P2PMappingSupported_GH100`：去掉 `!kbusIsStaticBar1Enabled` 的拒绝（动态不需要
  static BAR1）。顶层 `pcieP2PType==BAR1` 门禁本分支默认即满足（`kernel_bif.c:1182`），无需改。
- `kbusCreateP2PMappingForBar1P2P_GH100` / `Remove`：仅在两卡都启用 static BAR1 时建/拆 per-pair
  静态 IOMMU 映射；动态模式跳过（per-alloc IOMMU 在 nv_gpu_ops 做），只保留 refcount。
- `nvGpuOpsGetP2PCaps`：static 关时 `bar1DmaAddress/Size = 0` 且仍报 `p2pLink=PCIE_BAR1`，不调
  `kbusGetBar1P2PDmaInfo`（否则失败让 caps 查询挂掉）。

## UVM 兼容性结论（已核实）

UVM 的 `bar1_p2p_dma_base_address/size` **不在** GMMU-PTE 访问路径上（那条路径由我在
`nvGpuOpsBuildExternalAllocPtes` 直接编码），只被两处用：
- `uvm_gpu_peer_phys_address`（`uvm_gpu.c:3198`，CE **物理拷贝**寻址，加单一 base）——仅用于
  **UVM 托管内存（managed）的 peer 迁移**，NCCL 显式 P2P buffer 不走这条。
- `gpu_phys_address_is_bar1p2p_peer`（`:3300` 分类）——只在 `peer_copy_mode==VIRTUAL` 分支调
  （`:3316`）。
- peer identity 映射 `uvm_mmu_create_peer_identity_mappings` 在 `peer_copy_mode != VIRTUAL` 时
  直接 `return NV_OK`（`uvm_mmu.c:2415`）。

而 `peer_copy_mode` **默认 PHYSICAL**（`uvm_gpu.c:2043-2050`，除非 `uvm_peer_copy=virt`）。所以**默认
配置下**：peer identity 映射不建、bar1p2p 分类不调、`if(size) assert` 因 size=0 被跳过；`dma=0/0`
对 NCCL 的 VA 访问路径**无影响**。

**已知限制（默认配置外）**：
- 设 `uvm_peer_copy=virt` 会触发 peer identity 映射，按 `size`/`dma_base` 建一整块映射——动态模式下
  不成立。→ **保持默认 PHYSICAL**（别设 virt）。
- UVM **托管内存（cudaMallocManaged）的跨卡 CE 迁移**会用 `uvm_gpu_peer_phys_address`(+单一 base)，
  动态模式不支持 → 该路径地址错误（debug 构建会触发 `UVM_ASSERT(dma_size!=0)`）。NCCL 显式 buffer
  不受影响；但混用 managed memory P2P 的负载需注意。

## 当前状态（代码初步完成）

- `nv_gpu_ops.c`：自适应编码 + 动态窗口生命周期 + GetP2PCaps 修复 —— **已写完**。
- `kern_bus_gh100.c`：门禁解耦 + per-pair IOMMU 条件化 —— **已写完**。
- 均**未在编译器下验证**（macOS 无法编译）；上目标机后预期需要 compile-fix 迭代。
- 仍需在目标机核实：`memArea.pRanges` 由 `kbusUnmapFbAperture` 释放（无泄漏）；`kbusMapFbAperture`
  在 orchestrator 锁上下文可跑；NCCL 是否对 registered buffer 超 BAR1 预算优雅回退（默认 FIFO 集
  ~180MB 不触发）。预算准入（仿 `_isSpaceAvailableForBar1P2PMapping`）作为后续硬化项。
