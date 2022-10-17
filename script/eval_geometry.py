#!/usr/bin/env python

import os
import sys
import pathlib
import numpy as np
from seqName import stdres
from plot_geometry import PERIODIC, Data, read_dat
import multiprocessing


N_PROC = int(os.getenv("OMP_NUM_THREADS", "8"))
CHI_CUTOFF = 30.0
EPS = 1e-6


def kl_div(X, Y, eps=EPS):
    Y_nz = Y > eps
    return (Y[Y_nz] * np.log(Y[Y_nz] / np.maximum(eps, X[Y_nz]))).sum()


def assess_per_target(name, dat):
    delta_chi_s = [[] for _ in range(4)]
    #
    Y, X = dat
    for resName, y, x in zip(Y.resName, Y.c_ang, X.c_ang):
        mask = y < 999.0
        for i, m in enumerate(mask):
            if not m:
                break
            is_periodic = (resName, i + 1) in PERIODIC
            #
            delta = abs(x[i] - y[i])
            if is_periodic:
                delta %= 180.0
                delta = min(delta, abs(180.0 - delta))
            else:
                delta %= 360.0
                delta = min(delta, abs(360.0 - delta))
            delta_chi_s[i].append(delta)
    #
    rmsd = [np.sqrt(np.mean(np.power(X, 2))) for X in delta_chi_s]
    mae = [np.mean(X) for X in delta_chi_s]
    acc = [(np.array(X) < CHI_CUTOFF).astype(float).sum() / len(X) for X in delta_chi_s]
    #
    wrt = []
    for i in range(4):
        wrt.append(f"{rmsd[i]:6.2f} {mae[i]:6.2f} {acc[i]*100.0:6.2f}")
    wrt.append(name)
    return " | ".join(wrt) + "\n"


def distr_1d(xlim, bins, SS, native, model):
    Y, _ = np.histogram(native, bins, range=xlim, density=True)
    X, _ = np.histogram(model, bins, range=xlim, density=True)
    kl = kl_div(X, Y)
    return kl


def distr_omega(native, model):
    kl_s = {}
    for i, selection in enumerate([native.resName_next != "PRO", native.resName_next == "PRO"]):
        SS = native.select_by(selection).ss
        #
        native_sel = native.select_by(selection).t_ang[:, 2]
        native_sel[native_sel < -90] += 360.0
        model_sel = model.select_by(selection).t_ang[:, 2]
        model_sel[model_sel < -90] += 360.0
        n_residue = native_sel.shape[0]
        #
        for j, xlim in enumerate([(150, 210), (-30, 30)]):
            h_native, X = np.histogram(native_sel, 30, xlim, density=False)
            h_native = h_native.astype(float) / n_residue
            h_model, X = np.histogram(model_sel, 30, xlim, density=False)
            h_model = h_model.astype(float) / n_residue
            kl = kl_div(h_model, h_native)
            kl_s[(i, j, -1)] = kl
            #
            if i == 0 and j == 0:
                for ss, color in zip([0, 1, 2], ["red", "blue", "green"]):
                    subset = SS == ss
                    factor = len(SS) / (subset.astype(int).sum())
                    h_native, X = np.histogram(native_sel[subset], 30, xlim, density=False)
                    h_native = h_native.astype(float) / n_residue * factor
                    h_model, X = np.histogram(model_sel[subset], 30, xlim, density=False)
                    h_model = h_model.astype(float) / n_residue * factor
                    kl = kl_div(h_model, h_native)
                    kl_s[(i, j, ss)] = kl
    return kl_s


def distr_2d(native, model, periodic=False):
    if periodic:
        xylim = [[-180, 180], [-90, 90]]
    else:
        xylim = [[-180, 180], [-180, 180]]
    bins = 72
    dxy = (xylim[0][1] - xylim[0][0]) / bins
    #
    def process(_data):
        valid = np.all(_data[:, :2] < 360.0, axis=1)
        data = _data[valid]
        for k, xy in enumerate(xylim):
            dxy = xy[1] - xy[0]
            X = data[:, k]
            X[X > xy[1]] = X[X > xy[1]] - dxy
            X[X < xy[0]] = X[X < xy[0]] + dxy
            data[:, k] = X
        h = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=xylim, density=True)[0]
        return h

    #
    h_native = process(native)
    h_model = process(model)
    kl = kl_div(h_model, h_native)
    return kl


def assess_distr(data):
    wrt = ["bond_length: "]
    for i, xr in enumerate([(1.2, 1.5), (1.3, 1.6), (1.4, 1.7)]):
        kl = distr_1d(
            xr,
            60,
            data[0].ss,
            data[0].b_len[:, i],
            data[1].b_len[:, i],
        )
        wrt.append(f"{kl:10.3f}")
    sys.stdout.write(" ".join(wrt) + "\n")
    #
    wrt = ["bond_angle:  "]
    for i, xr in enumerate([(105, 135), (105, 135), (95, 125)]):
        kl = distr_1d(
            xr,
            60,
            data[0].ss,
            data[0].b_ang[:, i],
            data[1].b_ang[:, i],
        )
        wrt.append(f"{kl:10.3f}")
    sys.stdout.write(" ".join(wrt) + "\n")

    kl_s = distr_omega(data[0], data[1])
    wrt = ["omega_angle: "]
    for k, kl in kl_s.items():
        wrt.append(f"{kl:10.3f}")
    sys.stdout.write(" ".join(wrt) + "\n")

    wrt = ["rama_angles: "]
    kl = distr_2d(data[0].t_ang[:, :2], data[1].t_ang[:, :2])
    wrt.append(f"{kl:10.3f}")
    for i, name in enumerate(["helix", "sheet", "coil"]):
        select = data[0].ss == i
        kl = distr_2d(
            data[0].t_ang[select, :2],
            data[1].t_ang[select, :2],
        )
        wrt.append(f"{kl:10.3f}")
    sys.stdout.write(" ".join(wrt) + "\n")
    #
    for aa in stdres:
        if aa in ["ALA", "GLY"]:
            continue
        #
        select = data[0].resName == aa
        ss = data[0].ss[select]
        #
        wrt = [f"chi_s {aa}:   "]
        #
        kl = distr_1d(
            (-180, 180),
            60,
            None,
            data[0].c_ang[select, 0],
            data[1].c_ang[select, 0],
        )
        wrt.append(f"{kl:10.3f}")
        #
        for k in range(3):
            if not np.any(data[0].c_ang[select, k + 1] < 360.0):
                break
            kl = distr_2d(
                data[0].c_ang[select, k : k + 2],
                data[1].c_ang[select, k : k + 2],
                periodic=(aa, k + 2) in PERIODIC,
            )
            wrt.append(f"{kl:10.3f}")
        sys.stdout.write(" ".join(wrt) + "\n")


def main():
    log_dir = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2:
        keyword = sys.argv[2]
    else:
        keyword = "test"
    dat_fn_s = list(log_dir.glob(f"{keyword}*.geom.dat"))[:20]
    #
    data = [Data(), Data()]
    data[0].to_np()
    data[1].to_np()
    #
    n_proc = min(N_PROC, len(dat_fn_s))
    with multiprocessing.Pool(n_proc) as pool:
        dat_s = pool.map(read_dat, dat_fn_s)
    for dat in dat_s:
        for i in range(2):
            data[i].join(dat[i])
    assess_distr(data)

    with multiprocessing.Pool(n_proc) as pool:
        wrt = pool.starmap(
            assess_per_target, [(dat_fn.stem[:-5], dat) for dat_fn, dat in zip(dat_fn_s, dat_s)]
        )
    with open(f"{keyword}.chi_accuracy.dat", "wt") as fout:
        fout.writelines(wrt)


if __name__ == "__main__":
    main()
