# Environment inventory

> Fictional example. Capture at start, after every change, and attach to every result.

**System:** poc-node-01 (fictional) · **Captured:** 2026-11-02 09:30 · **By:** Alex Rivera (fictional)

| Layer | Item | Value |
|---|---|---|
| Hardware | Server model | OEM 2U server (fictional model X200) |
| Hardware | CPU | 2 x Intel Xeon (record exact SKU from ARK) |
| Hardware | Memory | 16 x 64 GB DDR5, all channels populated |
| Hardware | Accelerators | 4 x (record exact card and variant) |
| Hardware | NICs | 2 x 200 GbE (record model) |
| Firmware | BIOS | (version) |
| Firmware | Device firmware | (version per card) |
| OS | Distribution | Ubuntu 24.04.x (check support matrix) |
| OS | Kernel | (uname -r) |
| Driver/runtime | Device driver | (version) |
| Driver/runtime | Runtime libraries | (versions) |
| Framework | PyTorch / OpenVINO | (versions) |
| Serving | Engine and plugin | (vLLM version, plugin version) |
| Model | Name and revision | (model id + commit/hash) |
| Container | Image and digest | (registry/image@sha256:...) |
| Settings | Key env vars / flags | (list) |
| Topology | NUMA / pinning | (instances per socket, binding) |

**Deviations from official install guide:** none (or list them).
