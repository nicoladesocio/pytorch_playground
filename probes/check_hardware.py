"""Hardware / accelerator probe — run this BEFORE believing anything about your GPU.

    # inside the dev container (the full check)
    docker exec ppg-dev bash -lc 'cd /workspace && python probes/check_hardware.py'

    # on the host, to diagnose why the container cannot see the GPU
    python probes/check_hardware.py --host-only

WHY THIS EXISTS
---------------
HANDOFF_002 concluded "GPU is unreachable, run everything on CPU" and
`AGENT_INSTRUCTIONS.md` §4 promoted that into a standing rule. That conclusion was
**correct for that one machine** (WSL2, driver 465.89, GTX 960M `sm_50`) and wrong for
every other machine that has since run this repo. A hard-wired "no GPU" rule costs the
next agent on a real GPU days of CPU training for nothing.

So this probe measures the properties that actually decide GPU usability, and prints a
verdict. Nothing here is hardcoded per-device:

| Property                              | Why it decides usability                                                        |
|---------------------------------------|---------------------------------------------------------------------------------|
| driver version vs CUDA runtime        | too old a driver => `nvidia-smi` works but CUDA init fails                      |
| `nvidia-container-toolkit` installed  | absent => `docker run --gpus all` cannot work, regardless of the GPU            |
| `torch.cuda.get_arch_list()`          | the wheel's compiled arch list; if your GPU's `sm_XX` is absent, that wheel     |
                                       | cannot run on it no matter how new the driver is (this is what killed `sm_50`)  |
| real forward+backward on device       | the only proof that matters — kernel launch, cuDNN, and memory all working       |

Exit code is 0 if the verdict is GPU_OK or CPU_ONLY (both are actionable), 2 if the
GPU is present but unusable (i.e. there is a fixable setup problem worth chasing).
"""

import _bootstrap  # noqa: F401  (adds repo root to sys.path)

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_results = []


def record(status, check, detail=""):
    _results.append((status, check, detail))
    print(f"[{status:>4}] {check}" + (f" — {detail}" if detail else ""))


def _run(cmd, timeout=60):
    """Run a command, returning (rc, stdout+stderr). Never raises."""
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- host side
def check_host():
    print("\n--- host / driver ---")
    is_linux = sys.platform.startswith("linux")
    record(INFO, "platform", sys.platform)

    rc, out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader"])
    if rc != 0:
        record(FAIL, "nvidia-smi", out.strip().splitlines()[0] if out.strip() else "not available")
        gpus = []
    else:
        first = out.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        gpus = parts[0] if parts else "?"
        driver = parts[1] if len(parts) > 1 else "?"
        vram = parts[2] if len(parts) > 2 else "?"
        record(PASS, "nvidia-smi", f"{gpus}, driver {driver}, {vram}")
        try:
            major = int(re.match(r"(\d+)", driver).group(1))
        except Exception:
            major = 0
        # CUDA 12.x runtime needs driver >= 525; CUDA 11.x needs >= 450. Below 450
        # there is no usable CUDA on Linux at all (and WSL2 GPU passthrough did not
        # exist before 465) -- that is the class of failure HANDOFF_002 hit.
        if major < 450:
            record(FAIL, "driver version", f"{driver} predates every supported CUDA 12 runtime")
        elif major < 525:
            record(WARN, "driver version", f"{driver} is too old for CUDA 12.x wheels; "
                                           "expect CUDA init failures even if nvidia-smi works")
        else:
            record(PASS, "driver version", f"{driver} supports the CUDA 12.x runtime")

    print("\n--- container runtime ---")
    inside_container = os.path.exists("/.dockerenv")
    if inside_container:
        # The docker CLI is normally not installed inside the dev container, and every
        # check in this block is a host-side diagnostic. Report once and skip, so a
        # probe run inside the container does not spray three bogus FAILs.
        record(INFO, "container runtime",
               "running inside a container — host runtime checks skipped "
               "(run `python probes/check_hardware.py --host-only` on the host to see them)")
    else:
        rc, out = _run(["docker", "--version"])
        if rc != 0:
            record(FAIL, "docker", out.strip()[:120])
        else:
            record(PASS, "docker", out.strip())

    # nvidia-container-toolkit is what makes `--gpus all` work. Its absence is the
    # single most common reason a perfectly good GPU is invisible inside a container.
    # Both checks below are host-side, so they live inside the not-in-container branch.
    if not inside_container:
        rc, out = _run(["nvidia-ctk", "--version"])
        if rc != 0:
            rc2, out2 = _run(["dpkg-query", "-W", "nvidia-container-toolkit"])
            if rc2 == 0 and "install ok installed" in out2:
                record(PASS, "nvidia-container-toolkit", out2.strip().split()[1])
            else:
                record(FAIL, "nvidia-container-toolkit",
                       "not installed — `docker run --gpus all` cannot work. "
                       "Install it (see AGENT_INSTRUCTIONS.md §3.4) or use the venv path.")
        else:
            record(PASS, "nvidia-container-toolkit", out.strip().splitlines()[-1])

        rc, out = _run(["docker", "info", "--format", "{{.DefaultRuntime}} | {{json .Runtimes}}"])
        if rc == 0:
            has_nvidia = "nvidia" in out
            # Docker 25+ can also expose the GPU through an auto-injected CDI spec
            # instead of a registered runtime, so a bare `runc` here is not a failure:
            # Ubuntu's docker-packaged toolkit ships /etc/cdi/nvidia_com_gpu_*.json and
            # `--gpus all` still works. Checking only the runtime list produced a false
            # WARN on exactly such a host.
            rc2, out2 = _run("ls -1 /etc/cdi /var/run/cdi 2>/dev/null")
            cdi = out2.strip().replace("\n", " ")
            if has_nvidia or "nvidia" in cdi:
                record(PASS, "docker GPU access",
                       f"runtimes={out.strip()[:60]} cdi={cdi[:60] or '-'}")
            else:
                record(WARN, "docker GPU access",
                       f"no nvidia runtime/CDI found ({out.strip()[:80]}); `--gpus all` may fail")

    print("\n--- shared memory (DataLoader workers) ---")
    if is_linux:
        try:
            st = os.statvfs("/dev/shm")
            shm_gb = (st.f_bsize * st.f_blocks) / 1e9
            if shm_gb < 1:
                record(WARN, "/dev/shm", f"{shm_gb:.2f} GB — multi-worker DataLoaders will die; "
                                         "start the container with --shm-size=8g")
            else:
                record(PASS, "/dev/shm", f"{shm_gb:.1f} GB")
        except OSError as exc:
            record(INFO, "/dev/shm", f"unavailable ({exc})")
    else:
        record(INFO, "/dev/shm", "not a posix concept on this OS")

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    record(INFO, "CPUs visible", str(cpus))
    return gpus


# ------------------------------------------------------------------------ torch side
def check_torch():
    print("\n--- torch ---")
    try:
        import torch
    except ImportError as exc:
        record(FAIL, "import torch", f"{exc} — you are on the host interpreter; run inside the container")
        return None

    record(INFO, "torch", f"{torch.__version__} (CUDA built {torch.version.cuda})")
    try:
        import torchvision
        record(INFO, "torchvision", torchvision.__version__)
    except ImportError as exc:
        record(FAIL, "torchvision", str(exc))
    try:
        import lightning
        record(INFO, "lightning", lightning.__version__)
    except ImportError as exc:
        record(FAIL, "lightning", f"{exc} — pip install -e . was not run in this image")

    if not torch.cuda.is_available():
        # Distinguish "no GPU hardware" from "GPU present but torch cannot use it".
        rc, out = _run(["nvidia-smi", "-L"])
        if rc == 0 and "GPU 0" in out:
            record(FAIL, "torch.cuda.is_available()",
                   f"False although nvidia-smi reports a GPU ({out.strip().splitlines()[0]}). "
                   "This is a fixable setup problem: check the driver/CUDA version, the wheel's "
                   "arch list, and that the container was started with --gpus all.")
        else:
            record(WARN, "torch.cuda.is_available()",
                   "False and no GPU visible to nvidia-smi — this is a CPU-only host; "
                   "run the smoke tests on CPU and say so in your handoff")
        return None

    n = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1e9
    record(PASS, "torch.cuda", f"{n} device(s): {name}, sm_{cc[0]}{cc[1]}, {vram_gb:.1f} GB")

    # THE check that generalises HANDOFF_002's "sm_50 is unsupported" conclusion.
    # get_arch_list() is what the wheel was actually compiled for; if the device's
    # own architecture is missing from it, every kernel launch fails.
    arch_list = torch.cuda.get_arch_list()
    sm = f"sm_{cc[0]}{cc[1]}"
    if any(sm in a for a in arch_list) or "all" in arch_list or not arch_list:
        record(PASS, "wheel supports device arch", f"{sm} in {arch_list}")
    else:
        record(FAIL, "wheel supports device arch",
               f"{sm} NOT in {arch_list} — this torch build cannot run on this GPU. "
               "Install a torch build whose arch list covers it, or run on CPU.")

    # bf16/fp16 matter because train_final.py bakes in "16-mixed".
    try:
        has_bf16 = torch.cuda.is_bf16_supported()
    except Exception:
        has_bf16 = cc[0] >= 8
    record(INFO, "mixed precision", f"bf16={has_bf16} (needs sm_80+); fp16 available on all CUDA archs")

    if vram_gb < 4:
        record(WARN, "VRAM", f"{vram_gb:.1f} GB is tight — the baked-in batch sizes "
                             "(96/128 at 227px) will OOM; use the smoke sizes or lower batch_size")
    return {"name": name, "cc": cc, "vram_gb": vram_gb, "sm": sm}


def check_gpu_training():
    """Prove CUDA works for *this repo's* networks, not just for a matmul.

    A driver/wheel mismatch frequently passes a matmul smoke test (cublas is
    precompiled) and then dies on the first cuDNN conv. So the real gate is a
    forward + backward through AlexNet and GoogLeNet on the device.
    """
    import torch

    # 160px, not 96px: GoogLeNet's aux head AvgPool(kernel=5, stride=3) collapses
    # below ~128px (HANDOFF_003 §3.9 / P2-16), which would make this probe measure a
    # crippled branch and report a healthy network as broken.
    size = 160
    ok = True
    for label, factory in (("AlexNet", _make_alexnet), ("GoogLeNet", _make_googlenet)):
        try:
            net, criterion = factory()
            net = net.cuda().train()
            x = torch.randn(4, 3, size, size, device="cuda")
            y = torch.randint(0, 10, (4,), device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            loss = criterion(net(x), y) if label == "AlexNet" else criterion(net, x, y)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            grads = sum(1 for p in net.parameters() if p.grad is not None)
            total = sum(1 for _ in net.parameters())
            if not torch.isfinite(loss):
                record(FAIL, f"{label} gpu fwd/bwd", f"loss is {loss.item()}")
                ok = False
            else:
                record(PASS, f"{label} gpu fwd/bwd",
                       f"loss={loss.item():.4f} grads={grads}/{total} {dt:.2f}s "
                       f"(fp32, B=4, {size}px)")
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                record(WARN, f"{label} gpu fwd/bwd",
                       f"OOM at B=4/{size}px — GPU is too small for this repo's sizes; "
                       "use CPU or a smaller image_size")
            else:
                record(FAIL, f"{label} gpu fwd/bwd", msg.splitlines()[0][:200])
                ok = False
        except Exception as exc:
            record(FAIL, f"{label} gpu fwd/bwd", f"{type(exc).__name__}: {exc}")
            ok = False
    return ok


def _load_model_module(dir_name, module_name):
    """Import ``<repo>/<dir_name>/model.py`` as ``module_name``.

    Both networks' files are called ``model.py`` and both are imported as bare
    ``import model`` inside their own folder, so importing them into one process the
    obvious way silently reuses whichever was loaded first (this probe initially
    reported "module 'model' has no attribute 'AlexNet'" because GoogLeNet's
    ``model.py`` won the race). Loading each from its explicit path under a unique
    module name avoids the collision, and prepending its own folder to ``sys.path``
    keeps ``GoogLeNet/model.py``'s bare ``from inception import ...`` resolvable.
    """
    import importlib.util

    path = os.path.join(_bootstrap.REPO_ROOT, dir_name, "model.py")
    sys.path.insert(0, os.path.join(_bootstrap.REPO_ROOT, dir_name))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(os.path.join(_bootstrap.REPO_ROOT, dir_name))
    return module


def _make_alexnet():
    import torch.nn as nn
    mod = _load_model_module("AlexNet", "_probe_alexnet_model")
    return mod.AlexNet(num_classes=10, image_size=160), nn.CrossEntropyLoss()


def _make_googlenet():
    import torch.nn as nn
    mod = _load_model_module("GoogLeNet", "_probe_googlenet_model")
    net = mod.GoogLeNet(num_classes=10, image_size=160)
    ce = nn.CrossEntropyLoss()

    def criterion(net, x, y):
        # GoogLeNet returns (main, aux2, aux1) in training mode; mirror train.py's
        # combined loss so the number is comparable with the rest of the repo.
        out = net(x)
        if isinstance(out, (tuple, list)):
            main, aux1, aux2 = out[0], out[1], out[2]
            return ce(main, y) + 0.3 * ce(aux1, y) + 0.3 * ce(aux2, y)
        return ce(out, y)

    return net, criterion


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host-only", action="store_true",
                    help="skip the torch/CUDA checks (for diagnosing from the host interpreter)")
    ap.add_argument("--json", metavar="PATH", help="also write the results to a JSON file")
    args = ap.parse_args()

    print("=" * 72)
    print("pytorch_playground hardware probe")
    print("=" * 72)

    check_host()

    verdict, reasons = "CPU_ONLY", []
    if args.host_only:
        # Nothing was measured on the torch side, so "CPU_ONLY" would be a lie — the
        # host interpreter usually has no torch at all. Say what was actually checked.
        verdict, reasons = "HOST_ONLY", [
            "only host/driver/runtime checks ran (--host-only); no torch measurement",
            "run the probe inside the container for the GPU verdict",
        ]
    else:
        info = check_torch()
        if info is not None:
            gpu_ok = check_gpu_training()
            if gpu_ok:
                verdict, reasons = "GPU_OK", [
                    "CUDA forward+backward verified on both networks",
                    f"{info['name']} ({info['sm']}, {info['vram_gb']:.1f} GB)",
                ]
            else:
                verdict = "GPU_BROKEN"
                reasons = ["torch sees the GPU but a real conv run failed"]
        else:
            if any(s == FAIL and c.startswith("torch.cuda") for s, c, _ in _results):
                verdict, reasons = "GPU_BROKEN", ["nvidia-smi sees a GPU, torch does not"]

    print("\n" + "=" * 72)
    counts = {k: sum(1 for s, _, _ in _results if s == k) for k in (PASS, FAIL, WARN)}
    print(f"{PASS}={counts[PASS]}  {FAIL}={counts[FAIL]}  {WARN}={counts[WARN]}")
    print(f"VERDICT: {verdict}")
    for r in reasons:
        print(f"         {r}")
    if verdict == "GPU_OK":
        print("         => Run the smoke tests on GPU. Do NOT fall back to CPU and do NOT")
        print("            cite AGENT_INSTRUCTIONS.md's old 'GPU is unusable' note.")
    elif verdict == "CPU_ONLY":
        print("         => Run the smoke tests on CPU and record that this host has no GPU.")
    elif verdict == "HOST_ONLY":
        print("         => Inconclusive by construction. This mode only diagnoses the host;")
        print("            it cannot tell you whether training works.")
    else:
        print("         => There is a fixable setup problem. Fix it before falling back to")
        print("            CPU, and record what you found in your handoff.")
    print("=" * 72)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"verdict": verdict, "reasons": reasons,
                       "checks": [{"status": s, "check": c, "detail": d}
                                  for s, c, d in _results]}, fh, indent=2)
        print(f"wrote {args.json}")

    return 0 if verdict in ("GPU_OK", "CPU_ONLY", "HOST_ONLY") else 2


if __name__ == "__main__":
    sys.exit(main())
