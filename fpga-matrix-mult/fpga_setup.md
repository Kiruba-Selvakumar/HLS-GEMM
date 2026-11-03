# FPGA Setup Guide (IIIT-H)

This guide explains how to access IIIT FPGA nodes, configure the environment, and set up **Vitis** for U50 and VCK5000 boards.  
It is specific to the IIIT-H environment (VPN, RCE, and license server).

---

## Accessing RCE and Nodes

You need to be on the **IIIT intranet** or connected via **IIIT VPN**.

```bash
ssh username@rce
ssh username@node08   # from rce. (Note: node08 is not SLURM-managed)
```

---

## Vitis Installations

- **Vitis 2025.1**: `/home/Xilinx/2025.1/`  
- **Vitis 2022.2**: `/home/Xilinx/Vitis/2022.2/`

Usage guideline:
- **U50** → use **Vitis 2025.1**  
- **VCK5000** → use **Vitis 2022.2**

---

## Platform Files

All platform files are under:

```
/opt/xilinx/platforms/
```

- **VCK5000** → `xilinx_vck5000_gen4x8_qdma_2_202220_1`  
- **U50** → `xilinx_u50_gen3x16_xdma_5_202210_1`

You can verify device/platform details using:

```bash
xbutil examine
```

---

## Environment Setup

### For U50

Add the following to your `.bashrc`:

```bash
export LM_LICENSE_FILE=2100@rce.iiit.ac.in
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh
source /home/Xilinx/2025.1/Vitis/settings64.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platform
```

---

### For VCK5000

Add the following to your `.bashrc`:

```bash
source /home/Xilinx/Vitis/2022.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH
export LM_LICENSE_FILE=/home/Xilinx/Xilinx.lic
export PLATFORM_REPO_PATHS=/opt/xilinx/platform
```

---

## Notes

- Make sure you are connected to **VPN/intranet** before using the license server.  
- Keep `.bashrc` updated to avoid repeating environment setup every session.  
- `xbutil` is a useful tool to examine board status, memory, and health.  

---

[⬅ Back to Main README](./README.md)
