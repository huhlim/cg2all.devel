#!/usr/bin/env python

import os
import sys
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from seqName import stdres

PERIODIC = [("ASP", 2), ("GLU", 3), ("PHE", 2), ("TYR", 2)]


class Data(object):
    def __init__(self):
        self.has_chain_break = True
        self.chain_break = []
        self.resName = []
        self.ss = []
        self.asa = []
        self.b_len = []
        self.b_ang = []
        self.t_ang = []
        self.c_ang = []

    def __len__(self):
        return len(self.resName)

    def __repr__(self):
        return str(len(self))

    def append(self, x):
        self.chain_break.append(self.has_chain_break)
        self.has_chain_break = False
        #
        self.resName.append(x[2].strip().split()[-1])
        self.ss.append(["HELIX", "SHEET", "COIL"].index(x[3].strip().split()[0]))
        self.asa.append(x[3].strip().split()[1])
        self.b_len.append(x[4].split())
        self.b_ang.append(x[5].split())
        self.t_ang.append(x[6].split())
        self.c_ang.append(x[7].split())

    def append_chain_break(self):
        self.chain_break[-1] = True
        self.has_chain_break = True

    def to_np(self):
        self.resName_prev = np.array(self.resName[:-1])
        self.resName_next = np.array(self.resName[1:])
        #
        select = ~np.array(self.chain_break, dtype=bool)
        self.ss = np.array(self.ss, dtype=int)[select]
        self.asa = np.array(self.asa, dtype=int)[select]
        self.resName = np.array(self.resName)[select]
        self.resName_prev = self.resName_prev[select[:-1]]
        self.resName_next = self.resName_next[select[:-1]]
        self.b_len = np.array(self.b_len, dtype=float)[select]
        self.b_ang = np.array(self.b_ang, dtype=float)[select]
        self.t_ang = np.array(self.t_ang, dtype=float)[select]
        self.c_ang = np.array(self.c_ang, dtype=float)[select]

    def join(self, othr):
        if len(self) == 0:
            self.ss = othr.ss.copy()
            self.asa = othr.asa.copy()
            self.resName = othr.resName.copy()
            self.resName_prev = othr.resName_prev.copy()
            self.resName_next = othr.resName_next.copy()
            self.b_len = othr.b_len.copy()
            self.b_ang = othr.b_ang.copy()
            self.t_ang = othr.t_ang.copy()
            self.c_ang = othr.c_ang.copy()
        else:
            self.ss = np.concatenate([self.ss, othr.ss])
            self.asa = np.concatenate([self.asa, othr.asa])
            self.resName = np.concatenate([self.resName, othr.resName])
            self.resName_prev = np.concatenate([self.resName_prev, othr.resName_prev])
            self.resName_next = np.concatenate([self.resName_next, othr.resName_next])
            self.b_len = np.concatenate([self.b_len, othr.b_len])
            self.b_ang = np.concatenate([self.b_ang, othr.b_ang])
            self.t_ang = np.concatenate([self.t_ang, othr.t_ang])
            self.c_ang = np.concatenate([self.c_ang, othr.c_ang])

    def select_by(self, select):
        data = Data()
        data.ss = self.ss[select]
        data.asa = self.asa[select]
        data.resName = self.resName[select]
        data.resName_prev = self.resName_prev[select]
        data.resName_next = self.resName_next[select]
        data.b_len = self.b_len[select]
        data.b_ang = self.b_ang[select]
        data.t_ang = self.t_ang[select]
        data.c_ang = self.c_ang[select]
        return data


def join(dat_s):
    out = dat_s[0]
    out.ss = np.concatenate([x.ss for x in dat_s[1:]])
    out.asa = np.concatenate([x.asa for x in dat_s[1:]])
    out.resName = np.concatenate([x.resName for x in dat_s[1:]])
    out.resName_prev = np.concatenate([x.resName_prev for x in dat_s[1:]])
    out.resName_next = np.concatenate([x.resName_next for x in dat_s[1:]])
    out.b_len = np.concatenate([x.b_len for x in dat_s[1:]])
    out.b_ang = np.concatenate([x.b_ang for x in dat_s[1:]])
    out.t_ang = np.concatenate([x.t_ang for x in dat_s[1:]])
    out.c_ang = np.concatenate([x.c_ang for x in dat_s[1:]])
    return out


def read_dat(fn):
    data = [Data(), Data()]
    with open(fn) as fp:
        for line in fp:
            if line.startswith("#"):
                continue
            if line.strip() == "TER":
                data[model_no].append_chain_break()
            if "RESIDUE" not in line:
                continue
            x = line.strip().split(":")
            #
            model_no = int(x[0].split()[1])
            data[model_no].append(x)
    data[0].to_np()
    data[1].to_np()
    return data


def plot_1d(png_fn, xlim, bins, SS, native, model):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    #
    for marker, data in zip(["--", "-"], [native, model]):
        h, X = np.histogram(data, bins, range=xlim, density=True)
        X_cntr = 0.5 * (X[1:] + X[:-1])
        ax.plot(X_cntr, h, f"k{marker}", linewidth=2)
        #
        if SS is None:
            continue
        for ss, color in zip([0, 1, 2], ["red", "blue", "green"]):
            data_ss = data[SS == ss]
            h, X = np.histogram(data_ss, bins, range=xlim, density=True)
            X_cntr = 0.5 * (X[1:] + X[:-1])
            ax.plot(X_cntr, h, marker, color=color, linewidth=1)
    ax.set_xlim(xlim)
    ax.set_xticks(np.linspace(*xlim, 7))
    #
    fig.tight_layout()
    print(png_fn)
    plt.savefig(png_fn)
    plt.close("all")


def plot_omega(png_fn, native, model):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 4.8), sharey=False)
    #
    for i, selection in enumerate(
        [native.resName_next != "PRO", native.resName_next == "PRO"]
    ):
        SS = native.select_by(selection).ss
        #
        native_sel = native.select_by(selection).t_ang[:, 2]
        native_sel[native_sel < -90] += 360.0
        model_sel = model.select_by(selection).t_ang[:, 2]
        model_sel[model_sel < -90] += 360.0
        n_residue = native_sel.shape[0]
        #
        for j, xlim in enumerate([(150, 210), (-30, 30)]):
            ax = axes[i, j]
            #
            h_native, X = np.histogram(native_sel, 30, xlim, density=False)
            h_native = h_native.astype(float) / n_residue
            h_model, X = np.histogram(model_sel, 30, xlim, density=False)
            h_model = h_model.astype(float) / n_residue
            X_cntr = 0.5 * (X[1:] + X[:-1])
            #
            ax.plot(X_cntr, h_native, "--", color="black", linewidth=2)
            ax.plot(X_cntr, h_model, "-", color="black", linewidth=2)
            #
            if i == 0 and j == 0:
                for ss, color in zip([0, 1, 2], ["red", "blue", "green"]):
                    subset = SS == ss
                    factor = len(SS) / (subset.astype(int).sum())
                    h_native, X = np.histogram(
                        native_sel[subset], 30, xlim, density=False
                    )
                    h_native = h_native.astype(float) / n_residue * factor
                    h_model, X = np.histogram(
                        model_sel[subset], 30, xlim, density=False
                    )
                    h_model = h_model.astype(float) / n_residue * factor
                    ax.plot(X_cntr, h_native, "--", color=color, linewidth=1)
                    ax.plot(X_cntr, h_model, "-", color=color, linewidth=1)
            #
            ax.set_xlim(xlim)
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 7))
    #
    axes[0, 0].set_ylim((0, 0.3))
    axes[1, 0].set_ylim((0, 0.3))
    axes[0, 1].set_ylim((0, 0.015))
    axes[1, 1].set_ylim((0, 0.015))
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])
    #
    fig.tight_layout(h_pad=0.5)
    print(png_fn)
    plt.savefig(png_fn)
    plt.close("all")


def plot_2d(png_fn, native, model, kT=8, periodic=False):
    if periodic:
        xylim = [[-180, 180], [-90, 90]]
    else:
        xylim = [[-180, 180], [-180, 180]]
    bins = 72
    dxy = (xylim[0][1] - xylim[0][0]) / bins
    #
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharex=True, sharey=True)
    #
    for i, _data in enumerate([native, model]):
        valid = np.all(_data[:, :2] < 360.0, axis=1)
        data = _data[valid]
        for k, xy in enumerate(xylim):
            dxy = xy[1] - xy[0]
            X = data[:, k]
            X[X > xy[1]] = X[X > xy[1]] - dxy
            X[X < xy[0]] = X[X < xy[0]] + dxy
            data[:, k] = X

        h, X, Y = np.histogram2d(
            data[:, 0], data[:, 1], bins=bins, range=xylim, density=True
        )
        X = np.concatenate([[X[0] - dxy], X, [X[-1] + dxy]])
        Y = np.concatenate([[Y[0] - dxy], Y, [Y[-1] + dxy]])
        X_cntr = 0.5 * (X[1:] + X[:-1])
        Y_cntr = 0.5 * (Y[1:] + Y[:-1])
        xy = np.meshgrid(X_cntr, Y_cntr)
        #
        _fe = -np.log((h + 1e-10)).T
        _fe -= _fe.min()
        fe = np.zeros_like(xy[0], dtype=float)
        fe[1:-1, 1:-1] = _fe
        fe[0, :] = fe[-2, :]
        fe[-1, :] = fe[1, :]
        fe[:, 0] = fe[:, -2]
        fe[:, -1] = fe[:, 1]
        fe[0, 0] = fe[-2, -2]
        fe[-1, 0] = fe[1, -2]
        fe[0, -1] = fe[-2, 1]
        fe[-1, -1] = fe[1, 1]
        #
        axes[i].contourf(
            xy[0],
            xy[1],
            fe,
            cmap=plt.get_cmap("nipy_spectral"),
            levels=np.arange(0, kT, 0.5),
        )
        axes[i].contour(
            xy[0],
            xy[1],
            fe,
            colors="black",
            levels=np.arange(0, kT, 0.5),
            linewidths=0.2,
        )
    #
    for ax in axes:
        ax.set_xlim(xylim[0])
        ax.set_ylim(xylim[1])
        ax.set_xticks(np.linspace(*xylim[0], 7))
        ax.set_yticks(np.linspace(*xylim[1], 7))
    #
    fig.tight_layout()
    print(png_fn)
    plt.savefig(png_fn)
    plt.close("all")


def main():
    if len(sys.argv) == 1:
        log_dir = pathlib.Path(".")
    else:
        log_dir = pathlib.Path(sys.argv[1])
    if len(sys.argv) > 2:
        keyword = sys.argv[2]
    else:
        keyword = "test"
    dat_fn_s = log_dir.glob(f"{keyword}*.geom.dat")
    #
    data = [[Data()], [Data()]]
    data[0][0].to_np()
    data[1][0].to_np()
    #
    for dat_fn in list(dat_fn_s):
        dat = read_dat(dat_fn)
        for i in range(2):
            data[i].append(dat[i])
    data[0] = join(data[0])
    data[1] = join(data[1])
    #
    for i, xr in enumerate([(1.2, 1.5), (1.3, 1.6), (1.4, 1.7)]):
        plot_1d(
            log_dir / f"{keyword}.b_len{i}.png",
            xr,
            60,
            data[0].ss,
            data[0].b_len[:, i],
            data[1].b_len[:, i],
        )
    for i, xr in enumerate([(105, 135), (105, 135), (95, 125)]):
        plot_1d(
            log_dir / f"{keyword}.b_ang{i}.png",
            xr,
            60,
            data[0].ss,
            data[0].b_ang[:, i],
            data[1].b_ang[:, i],
        )

    plot_omega(log_dir / f"{keyword}.omega.png", data[0], data[1])

    plot_2d(
        log_dir / f"{keyword}.rama_all.png", data[0].t_ang[:, :2], data[1].t_ang[:, :2]
    )
    for i, name in enumerate(["helix", "sheet", "coil"]):
        select = data[0].ss == i
        plot_2d(
            log_dir / f"{keyword}.rama_{name}.png",
            data[0].t_ang[select, :2],
            data[1].t_ang[select, :2],
        )
    #
    for aa in stdres:
        if aa in ["ALA", "GLY"]:
            continue
        #
        select = data[0].resName == aa
        ss = data[0].ss[select]
        #
        plot_1d(
            log_dir / f"{keyword}.chi_1.{aa}.png",
            (-180, 180),
            60,
            None,
            data[0].c_ang[select, 0],
            data[1].c_ang[select, 0],
        )
        #
        for k in range(3):
            if not np.any(data[0].c_ang[select, k + 1] < 360.0):
                break
            plot_2d(
                log_dir / f"{keyword}.chi_{k+2}.{aa}.png",
                data[0].c_ang[select, k : k + 2],
                data[1].c_ang[select, k : k + 2],
                kT=6,
                periodic=(aa, k + 2) in PERIODIC,
            )


if __name__ == "__main__":
    main()
