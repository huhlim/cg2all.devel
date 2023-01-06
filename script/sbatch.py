#!/usr/bin/env python

import os
import sys
import pathlib
import argparse
import subprocess as sp
from tempfile import TemporaryFile

import libslurm


def get_dependency(keyword):
    lines = (
        sp.check_output(["squeue", "-h", "-u", "huhlim", "-o", "%i %P %j %T %R"])
        .decode("utf8")
        .split("\n")[:-1]
    )
    #
    selected = []
    for line in lines:
        job_id, partition, name, state, node = line.strip().split(maxsplit=4)
        if keyword in name:
            selected.append(job_id)
    return selected


def sbatch(queue, dep_s=[], hold=False):
    cmd = ["sbatch"]
    if hold:
        cmd.append("--hold")
    if len(dep_s) > 0:
        cmd.append("--dependency=%s" % (",".join([f"afterany:{dep}" for dep in dep_s])))
    #
    stdin = TemporaryFile(mode="w+t")
    stdin.write(queue)
    stdin.flush()
    stdin.seek(0)
    #
    output = sp.check_output(cmd, stdin=stdin)
    output = output.decode("utf-8").strip().split()[-1]
    return [output]


def main():
    arg = argparse.ArgumentParser(prog="sbatch.script")
    arg.add_argument(dest="config_json_fn")
    #
    arg.add_argument("--name", dest="name", default=None)
    arg.add_argument("--gpu", dest="n_gpu", default=2, type=int)
    arg.add_argument("--cpu", dest="n_cpu", default=8, type=int)
    arg.add_argument("--hold", dest="hold", default=False, action="store_true")
    #
    arg.add_argument("--test", dest="n_test", default=1, type=int)
    arg.add_argument("--epoch", dest="n_epoch", default=100, type=int)
    arg.add_argument("--ckpt", dest="ckpt", default=None)
    arg.add_argument("--dep", dest="has_dep", default=False, action="store_true")
    arg.add_argument(
        "--continue", dest="continue_run", default=False, action="store_true"
    )
    #
    arg.add_argument("--exec", dest="EXEC", default="./script/train.py")
    #
    if len(sys.argv) == 1:
        return arg.print_help()
    #
    arg = arg.parse_args()
    #
    arg.config_json_fn = pathlib.Path(arg.config_json_fn)
    if arg.name is None:
        arg.name = arg.config_json_fn.stem
    #
    output_dir = pathlib.Path(f"lightning_logs/{arg.name}")
    prev_s = list(output_dir.glob("version_*"))
    prev_s.sort(key=lambda x: int(x.name.split("_")[-1]))
    #
    if arg.continue_run and arg.ckpt is None:
        arg.ckpt = prev_s[-1] / "last.ckpt"
    #
    if arg.has_dep:
        dep_s = get_dependency(arg.name)
    else:
        dep_s = []
    #
    version = len(prev_s)
    for k in range(arg.n_test):
        cmd = [arg.EXEC]
        cmd.extend(["--name", arg.name])
        cmd.extend(["--config", str(arg.config_json_fn)])
        cmd.extend(["--epoch", str(arg.n_epoch)])
        #
        if arg.ckpt is not None:
            cmd.extend(["--ckpt", str(arg.ckpt)])
        cmd = " ".join(cmd)
        print(cmd)
        #
        cmd = [cmd]
        cmd.append(f"cd {str(output_dir)}/version_{version}")
        cmd.append(f"molprobity.py -np {arg.n_cpu} test_*.pdb")
        cmd.append("calc_geometry.py test_*.pdb")
        cmd.append(f"{os.environ['work']}/ml/cg2all/script/plot_geometry.py")
        cmd.append(f"{os.environ['work']}/ml/cg2all/script/eval_geometry.py")
        cmd.append(f"{os.environ['work']}/ml/cg2all/script/eval_similarity.py")
        #
        log_fn = f"logs/{arg.name}.version_{version}.log"
        queue = libslurm.write(
            cmd,
            name=f"{arg.name}.v{version}",
            conda="dgl",
            n_gpu=arg.n_gpu,
            n_cpu=arg.n_cpu,
            output=log_fn,
        )
        #
        dep_s = sbatch(queue, dep_s=dep_s, hold=arg.hold)
        #
        arg.ckpt = output_dir / f"version_{version}/last.ckpt"
        version += 1


if __name__ == "__main__":
    main()
