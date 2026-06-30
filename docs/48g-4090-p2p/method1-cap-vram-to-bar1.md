# 方案一：把可用显存裁到 BAR1 以内，让 static BAR1 P2P 原样工作

> 这是 48GB 4090（AD102 + 泄漏 VBIOS + 3090Ti PCB，**32GB BAR1 / 48GB 显存**）解锁 PCIe P2P
> 的几个候选方向之一。本方案**能跑通**，但代价是损失约 18GB 显存。记录在此作为保底方案；
> 保留全部 48GB 的方向另行调查。

## 1. 问题本质：BAR1 P2P 硬依赖 "BAR1 ≥ 显存"

本仓库的 P2P mod 走的是 **GH100 的 static BAR1 P2P** 这条路（不是老的 mailbox）。它把 peer 显存
当成"挂在 peer BAR1 物理基址上的一段 sysmem"来访问：

- 改过的 `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c` 里，peer 映射每个 PTE 的物理地址 =
  `bar1BusAddr + fbPhysOffset`，其中 `bar1BusAddr = gpumgrGetGpuPhysFbAddr(peerGpu)`（peer 的
  BAR1 PCIe 基址），并把 `GMMU_APERTURE_PEER` 改写成 `SYS_NONCOH`（配合 `gmmu_fmt.c:124` 把 PEER
  aperture 的地址域指向 `fldAddrSysmem`）。
- 因此**只有当 `fbPhysOffset < BAR1_size` 时，`bar1BusAddr + fbPhysOffset` 才落在 peer 的 BAR1
  窗口内、才指向正确的显存**。一旦某块显存的物理偏移 ≥ 32GB，编码出来的总线地址就越过了 BAR1
  窗口，访问会损坏数据或挂死。

而 BAR1 P2P 整条链硬依赖 static BAR1 被启用：

```
kern_bus_gh100.c:1453  kbusIsPcieBar1P2PMappingSupported_GH100
    if (!kbusIsStaticBar1Enabled(pGpu0, ...) || !kbusIsStaticBar1Enabled(pGpu1, ...))
        return NV_FALSE;
```

static BAR1 又硬要求 **BAR1 ≥ client 可见 FB**，写死、无 regkey 可绕
（`src/nvidia/src/kernel/gpu/bus/arch/turing/kern_bus_tu102.c` `kbusIsStaticBar1Supported_TU102`）：

- `FORCE_STATIC_BAR1_ENABLE`（`tu102.c:432`）：
  `if (bar1VASizeAligned < bar1MapSize) { DBG_BREAKPOINT(); return NV_ERR_INVALID_REGISTRY_KEY; }`
  —— `bar1MapSize` 就是 client FB。
- `AUTO`（`tu102.c:498`）：要求 `bar1VASizeAligned >= fbSizeAligned + userD + doorbell + console
  + mailbox + 512MB padding`，比全 FB 还多。

**结论**：32GB BAR1 / 48GB FB → static BAR1 在任何模式都拿不到 →
`kbusCreateP2PMappingForBar1P2P_GH100` 返回 `NV_ERR_NOT_SUPPORTED`，加上 mod 已把 `forceP2PType`
钉死成 BAR1P2P、无回退路径，所以 NCCL 在建链阶段卡死。这正是实测"卡在初始化"的精确原因。
正常 4090（24/24）和 RTX 5090（32/32）都满足 BAR1 ≥ FB，所以能用。

## 2. 思路：把这张卡伪装成 5090

既然 5090（32GB BAR1 + 32GB 显存）能正常工作，那就把这张卡对外呈现的**可分配显存裁到 32GB BAR1
以内**（再留出一点 overhead），static BAR1 自然就能开，现有 P2P mod 一行不改即可工作。代价是顶部
约 18GB 显存变为不可用。

关键点：**本机访问自己显存不经过 BAR1**，只有 P2P 才需要 BAR1。所以"砍掉"的其实只是 P2P 可达性，
但由于这里是整体裁 PMA，本机也用不到那部分了——这是本方案的主要代价。

## 3. 为什么纯靠 regkey 就能实现（已逐条核对代码）

无需改任何源码，两个 NVIDIA 官方支持的 regkey 即可：

- **`OverrideFbSize`**（MB）→ `Ram.fbOverrideSizeMb`（`mem_mgr.c:187`），由
  `memmgrHandleSizeOverrides_GP100`（`mem_mgr_gp100.c:79`）执行：当总显存 > override 时，
  **在可用区顶部插入一段 `(fbTotal − override)` 大小的 reserved region**，把它从可分配池移除。
  已确认该 HAL 对所有桌面 dGPU（含 AD102 / GB202 / GH100）生效（`g_mem_mgr_nvoc.c:758`，
  只有 Tegra 用 stub）。
- PMA 注册时跳过 reserved region，所以 `pmaGetClientAddrSpaceSize` = 最高**可用** region limit + 1
  （`phys_mem_allocator.c:1933`）。裁掉可用区顶部后，`memmgrGetClientFbAddrSpaceSize` 随之降到
  32GB 以下。static BAR1 映射的也正是 `[0, clientFB)` 这段（`kern_bus_tu102.c:551-562`），
  且 PMA 只在这段内分配，P2P 编码 `bar1BusAddr + fbOffset` 全程合法。
- **`RMForceStaticBar1=1`**（`NV_REG_STR_RM_FORCE_STATIC_BAR1_ENABLE`，`kern_bus.c:189` 读取）
  **必须设**：`AUTO` 检查用的是 `Ram.fbAddrSpaceSizeMb`（仍约 48GB，因为 override 只动可用区、
  没改顶部 GSP reserved region 的 limit），所以 AUTO 仍然不过；`FORCE_ENABLE` 只比 client FB
  （现在约 30GB ≤ 32GB）→ 通过。

时序也对：PMA / region 在 MemoryManager **StateInit** 建好
（`memmgrStateInitLocked → memmgrCreateHeap`，`mem_mgr.c:657,1079,1111`），static BAR1 在更晚的
KernelBus **StatePostLoad** 才启用（`kern_bus_gm107.c:702`），看到的是裁过的 size。

## 4. 具体配置

在**目标 Debian 主机**上加 modprobe 配置（不改源码）：

```
# /etc/modprobe.d/nvidia-48g-p2p.conf
options nvidia NVreg_RegistryDwords="OverrideFbSize=30720;RMForceStaticBar1=1"
```

- `OverrideFbSize` 单位 **MB**，`30720`（=30GB）是个安全起点。static BAR1 区起始于
  `bar1Offset = align_up(console+mailbox, 512MB)`（约 512MB），所以需 `clientFB + ~512MB ≤ 32GB`；
  30GB 余量充足。想再榨一点可往 ~31744 调，但每次都要验证 static BAR1 仍能启用。
- 之后用现有 `install.sh` 重新编译加载（DKMS 包流程则 `update-initramfs` + 重启）。
- **混插不同型号卡的主机**：用 `NVreg_RegistryDwordsPerDevice="pci=DDDD:BB:DD.F;OverrideFbSize=
  30720;RMForceStaticBar1=1"` 把裁切只作用在 48GB 卡上，别误伤正常卡（注：对 ≤30GB 的卡
  `OverrideFbSize=30720` 是 no-op；但 5090 会被砍掉 2GB，所以混插时务必按设备作用域）。

## 5. 验证清单（在目标主机上，按 GPU 对）

1. **裁切生效**：`nvidia-smi` → 单卡总显存 ≈ **30GB**（从 48GB 降下来）。若仍 48GB，说明 regkey
   没被解析（查 `/proc/driver/nvidia/params`）。
2. **static BAR1 / P2P 开启**：`nvidia-smi topo -p2p r` 和 `-p2p w` → 卡间显示 `OK`。
   `dmesg` 不应出现 "BAR1 size … is not large enough"（出现 = 裁得还不够 → 调小 `OverrideFbSize`）。
3. **功能性 P2P**：CUDA 的 `p2pBandwidthLatencyTest` / `simpleP2P` 通过，且带宽是 P2P 级别
   （不是绕 sysmem 的数字）。
4. **NCCL 端到端**：`nccl-tests` 的 `all_reduce_perf -g 2` 在两卡上不卡 init，带宽符合 PCIe P2P。

## 6. 代价与限制

- **损失约 18GB 显存** —— 把 FB 塞进 32GB BAR1 的固有代价，也是要继续找其他方案的原因。
  好处是对用户态诚实（CUDA 看到 ~30GB，不会出现超出上限的意外 OOM）。
- release 驱动里 `INFO`/`WARNING` 级 `NV_PRINTF`（如 "Static bar1 mapped …"、"added PCIe BAR1
  P2P mapping"）被编译掉了，验证以**功能性测试**为准；失败时的 `LEVEL_ERROR`
  "not large enough" 那行**会**打印。
- 每次驱动版本升级后需重新验证（size-override 与 static-BAR1 路径稳定，但 mod 本身跟随版本）。

## 7. 结论

- **可行、低风险、零源码改动**，是保底方案。
- 唯一硬伤：**18GB 显存浪费**。若工作负载本就吃不到 48GB（NCCL 通信 buffer 通常不大），此方案
  足够；若要保住全部显存，见后续方案调查。

## 附：为什么不能用别的现成路径（背景）

- **动态 BAR1**（`kbusMapFbAperture`，`kern_bus_gm107.c`）：确实是"BAR1 装不下显存"年代的按需
  pin+map+建 GMMU 页表机制，但它服务的是**本机 CPU/引擎**映射，**不暴露给 peer 做 P2P**；而且
  VA 用满即失败、**无换出**，不是 pager。
- **Mailbox P2P**：历史上"BAR1 < FB 也能 P2P"的真正答案，但 Ada 之后硬件被砍、默认关闭
  （`pcieP2PType` 默认 BAR1），且开源树里连寄存器定义（`NV_XAL_EP_P2P_WMBOX_ADDR_ADDR`）都缺失，
  带宽也差。这正是社区当年放弃 mailbox 转投 BAR1 identity 的原因。
- **透明分页 BAR1 P2P**（理想中的"正道"）：硬件层面做不出来——peer 发来的 inbound BAR1 访问触发的
  是**不可重放**的 GMMU/PRI fault（写是 posted 丢了就丢、读有 PCIe completion 超时），无法
  "fault→换页→重试"。所以驱动里没有、也不可能有这段代码。
