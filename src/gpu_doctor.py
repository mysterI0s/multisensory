#!/usr/bin/env python3
"""gpu_doctor.py -- find out exactly why TensorFlow cannot see the GPU.

Does NOT import tensorflow until the very end, so it still works when the
TF GPU stack is broken. Probes every CUDA shared library directly with
dlopen and reports which ones the dynamic loader can and cannot find.

    python gpu_doctor.py            # diagnose
    python gpu_doctor.py --fix      # diagnose, then print the export line
"""
import argparse
import ctypes
import os
import sys
import glob

# Libraries TensorFlow dlopens for a CUDA 12 build. The driver library
# (libcuda) comes from the WSL projection; the rest come from pip wheels.
REQUIRED = [
    ("libcuda.so.1", "driver (WSL projects this from Windows)"),
    ("libcudart.so.12", "cuda runtime"),
    ("libcublas.so.12", "cublas"),
    ("libcublasLt.so.12", "cublasLt"),
    ("libcufft.so.11", "cufft"),
    ("libcurand.so.10", "curand"),
    ("libcusolver.so.11", "cusolver"),
    ("libcusparse.so.12", "cusparse"),
    ("libcudnn.so.9", "cudnn"),
    ("libnvJitLink.so.12", "nvjitlink"),
    ("libnccl.so.2", "nccl (optional, single GPU does not need it)"),
]

WSL_LIB = "/usr/lib/wsl/lib"


def hr(title):
    print("")
    print("=" * 68)
    print(title)
    print("=" * 68)


def probe(name):
    """Try to dlopen by bare name, using the loader's current search path."""
    try:
        ctypes.CDLL(name)
        return True, None
    except OSError as e:
        return False, str(e)


def find_nvidia_lib_dirs():
    """Locate the lib/ directories inside the installed nvidia-*-cu12 wheels."""
    dirs = []
    try:
        import nvidia
    except ImportError:
        return dirs
    roots = list(getattr(nvidia, "__path__", []))
    for root in roots:
        for sub in sorted(glob.glob(os.path.join(root, "*", "lib"))):
            if os.path.isdir(sub) and glob.glob(os.path.join(sub, "*.so*")):
                dirs.append(sub)
    return dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="print the export line that repairs the search path")
    ap.add_argument("--tf", action="store_true",
                    help="import tensorflow at the end and list devices")
    args = ap.parse_args()

    hr("interpreter")
    print("executable      : %s" % sys.executable)
    print("version         : %s" % sys.version.split()[0])
    print("sys.prefix      : %s" % sys.prefix)
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    print("inside venv     : %s" % in_venv)
    if not in_venv:
        print("  (!) Not in a virtualenv. If you installed TensorFlow into")
        print("      a venv, you are running the WRONG python.")

    hr("environment")
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    print("LD_LIBRARY_PATH : %s" % (cur if cur else "(empty)"))
    if WSL_LIB not in cur.split(":"):
        print("  (!) %s is NOT on the path." % WSL_LIB)
    print("TF_USE_LEGACY_KERAS : %s"
          % os.environ.get("TF_USE_LEGACY_KERAS", "(unset)"))

    hr("wsl driver projection")
    if os.path.isdir(WSL_LIB):
        hits = sorted(glob.glob(os.path.join(WSL_LIB, "libcuda.so*")))
        for h in hits:
            print("  found %s" % h)
        if not hits:
            print("  (!) no libcuda.so* in %s" % WSL_LIB)
    else:
        print("  (!) %s does not exist. Is this really WSL2?" % WSL_LIB)

    hr("nvidia pip wheel lib directories")
    nvdirs = find_nvidia_lib_dirs()
    if not nvdirs:
        print("  (!) No nvidia wheel lib dirs found. The [and-cuda] extra")
        print("      did not install, or you are in the wrong environment.")
    for d in nvdirs:
        n = len(glob.glob(os.path.join(d, "*.so*")))
        print("  %-58s %3d .so" % (d.replace(sys.prefix, "$VENV"), n))

    hr("dlopen probe (current search path)")
    missing = []
    for name, desc in REQUIRED:
        ok, err = probe(name)
        print("  %-22s %-4s  %s" % (name, "OK" if ok else "FAIL", desc))
        if not ok:
            missing.append((name, err))

    if missing:
        hr("failures in detail")
        for name, err in missing:
            print("  %s" % name)
            print("    %s" % err)

    # Can the files be loaded at all, given an explicit full path?
    hr("locating the missing files on disk")
    search = nvdirs + [WSL_LIB]
    for name, _ in missing:
        found = []
        for d in search:
            p = os.path.join(d, name)
            if os.path.exists(p):
                found.append(p)
        if found:
            for p in found:
                try:
                    ctypes.CDLL(p)
                    verdict = "loads OK by full path"
                except OSError as e:
                    verdict = "present but FAILS to load: %s" % e
                print("  %-22s -> %s" % (name, verdict))
                print("      %s" % p)
        else:
            print("  %-22s -> NOT FOUND anywhere in the search dirs" % name)

    hr("verdict")
    hard = [m for m, _ in missing if "nccl" not in m]
    if not missing:
        print("  All libraries load. If TensorFlow still reports no GPU,")
        print("  the problem is not the dynamic loader.")
    elif not hard:
        print("  Only nccl is missing, which a single-GPU setup does not")
        print("  need. This is not what is blocking you.")
    else:
        print("  %d required libraries cannot be dlopened." % len(hard))
        print("  If they showed 'loads OK by full path' above, this is a")
        print("  search-path problem and --fix will solve it.")

    if args.fix or missing:
        parts = nvdirs + [WSL_LIB]
        line = "export LD_LIBRARY_PATH=%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
               % ":".join(parts)
        hr("suggested fix")
        print("Run this, then retest in the SAME shell:")
        print("")
        print(line)
        print("")
        print("To make it stick, append it to the venv activate script so it")
        print("only applies inside this project:")
        print("")
        print("  echo '%s' >> %s/bin/activate" % (line, sys.prefix))

    if args.tf:
        hr("tensorflow")
        try:
            import tensorflow as tf
            print("version : %s" % tf.__version__)
            print("GPUs    : %s" % tf.config.list_physical_devices("GPU"))
        except Exception as e:
            print("import failed: %s" % e)


if __name__ == "__main__":
    main()

