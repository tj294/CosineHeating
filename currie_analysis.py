"""
Analysis scripts for cosine heated cases (doesn't average over heating width)
Will calculate:
    - time tracks for KE, Nu and Re
    - Flux profiles
    - Temp profiles
    - Nusselt number
    - Reynolds number
    - Rossby number
    - <TH> and <(grad(T))^2>
    - Rf<wT> and <(grad(u))^2>

Usage:
    currie_analysis.py <infile> [options]

Options:
    --ASI [TIME]        # Time to start averaging from [default: -1]
    --AD [TIME]         # Duration of averaging window [default: 2.0]
    --window [WINDOW]   # Window for rolling average [default: 0.05]
    --Hwidth [Hwidth]   # Width of heating layers [default: 0.2]
    -e                  # Energy tracks
    -f                  # Fluxes
    -t                  # Temperature profiles
    -n                  # Nusselt Number
    -v                  # Reynolds number
    -r                  # Rossby number
    -s                  # Power integrals
    -a                  # All quantities
    --gif               # Create temp field gif for averaging time
    -h --help           # Show this screen
    --version           # Show version
"""

from docopt import docopt
import h5py as h5
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import interp2d
import json, re
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import dedalus.public as d3
import imageio.v2 as iio

def combine_array(small_arr, big_field, small_y, small_z, axis=0):
    # Check spatial shapes are the same (Dim 0 is time)
    big_arr = np.array(big_field)[:, 0, :, :]
    if small_arr.shape[-2:] == big_arr.shape[-2:]:
        outarr = np.concatenate((small_arr, big_arr), axis=axis)
        return outarr, small_y, small_z
    else:
        print("\tInterpolating Field")
        big_arr = np.array(big_field)[:, 0, :, :]
        # Rescale arr1 onto targetShape
        big_y = np.array(big_field.dims[2]['y'])
        big_z = np.array(big_field.dims[3]['z'])
        timelen = small_arr.shape[0]
        reshapeSpatial = big_arr.shape[1:]
        reShape = (timelen, *reshapeSpatial)
        reShape_arr = np.zeros(reShape)
        for i in range(timelen):
            interp = interp2d(small_y, small_z, small_arr[i].T)
            reShape_arr[i] = interp(big_y, big_z).T
        out_arr = np.concatenate((reShape_arr, big_arr), axis=0)
        return (out_arr, big_y, big_z)

def get_ave_indices(time):
    if float(args["--ASI"]) < 0:
        AEI = -1
        ASI = np.abs(time - (time[AEI] - float(args["--AD"]))).argmin()
    else:
        ASI = np.abs(time - float(args["--ASI"])).argmin()
        AEI = np.abs(time - (time[ASI] + float(args["--AD"]))).argmin()
    return ASI, AEI


def rolling_average(quantity, time, window: float):
    assert len(time) == len(quantity)
    run_ave = []
    for i, t0 in enumerate(time):
        mean = np.nanmean(
            quantity[(time > t0 - window / 2) & (time <= t0 + window / 2)]
        )
        run_ave.append(mean)
    return np.array(run_ave)


def get_heat_func():
    try:
        snaps = list(direc.glob("snapshots/snapshots_s*.h5"))
        snaps.sort(key=lambda f: int(re.sub("\D", "", str(f))))
        with h5.File(snaps[-1], "r") as file:
            heat_func = np.array(file["tasks"]["heat"])[0, 0, 0, :]
    except:
        print(f"Heat Func not found.\nWriting heat function.")
        H = 1
        F = 1
        heating_width = float(args["--Hwidth"])
        L = z[-1]
        # Width of heating and cooling layers
        Delta = heating_width
        print(L, Delta)
        heat_func = lambda z: (F / Delta) * (
            1 + np.cos((2 * np.pi * (z - (Delta / 2))) / Delta)
        )
        cool_func = lambda z: (F / Delta) * (
            -1 - np.cos((2 * np.pi * (z - L + (Delta / 2))) / Delta)
        )

        heat_func = np.piecewise(
            z, [z <= Delta, z >= L - Delta], [heat_func, cool_func, 0]
        )
    return heat_func


args = docopt(__doc__, version="1.0")
fluxes = args['-f']
nusselt = args['-n']
temp_profile = args['-t']
energies = args['-e']
power_integrals = args['-s']
reynolds = args['-v']
rossby = args['-r']

if args['-a']:
    fluxes = nusselt = temp_profile = energies = power_integrals = reynolds = rossby = True

# print(args)
containCZ = True
direc = Path(args["<infile>"])
if not direc.exists():
    raise FileNotFoundError(f"Directory {direc} does not exist")
outpath = direc.joinpath("images/")
outpath.mkdir(exist_ok=True)
outvalues = {}

#! Params
with open(direc.joinpath("run_params/runparams.json"), 'r') as f:
    params = json.load(f)
Rf = params['Ra']
Pr = params['Pr']
Ta = params['Ta']
Ly, Lz = params['Ly'], params['Lz']
Ny, Nz = params['Ny'], params['Nz']

outvalues['Rf'] = Rf
outvalues['Pr'] = Pr
outvalues['Ta'] = Ta

#! Time Tracks
if energies:
    scalar_files = list(direc.glob("scalars/scalars_s*.h5"))
    scalar_files.sort(key=lambda f: int(re.sub("\D", "", str(f))))
    print("Loading scalars...")
    for i, sc_file in enumerate(scalar_files):
        if i == 0:
            with h5.File(sc_file, "r") as file:
                sc_time = np.array(file["scales"]["sim_time"])
                Re = np.array(file["tasks"]["Re"])
                KE = np.array(file["tasks"]["KE"])
                try:
                    scNu = np.array(file["tasks"]["Nu_inst"])
                    scNu_cz = np.array(file["tasks"]["Nu_cz_inst"])
                    RfwT_cz = np.array(file["tasks"]["Ra*<wT>_cz"])
                    qT_cz = np.array(file["tasks"]["<QT>_cz"])
                    gradu_cz = np.array(file["tasks"]["<(grad u)^2>_cz"])
                    gradT_cz = np.array(file["tasks"]["<(grad T)^2>_cz"])
                    Ro = np.array(file["tasks"]["Ro"])
                    Ro_cz = np.array(file["tasks"]["Ro_cz"])
                    Re_cz = np.array(file["tasks"]["Re_cz"])
                except:
                    print("CZ Vals not available")
                    containCZ = False
        else:
            with h5.File(sc_file, "r") as file:
                sc_time = np.concatenate(
                    (sc_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                Re = np.concatenate((Re, np.array(file["tasks"]["Re"])), axis=0)

                KE = np.concatenate((KE, np.array(file["tasks"]["KE"])), axis=0)
                # gradT = np.concatenate(
                #     (gradT, np.array(file["tasks"]["<(grad T)^2>"])), axis=0
                # )
                # gradu = np.concatenate(
                #     (gradu, np.array(file["tasks"]["<(grad u)^2>"])), axis=0
                # )
                if containCZ:
                    scNu = np.concatenate(
                        (scNu, np.array(file["tasks"]["Nu_inst"])), axis=0
                    )
                    Ro = np.concatenate((Ro, np.array(file["tasks"]["Ro"])), axis=0)
                    scNu_cz = np.concatenate(
                        (scNu_cz, np.array(file["tasks"]["Nu_cz_inst"])), axis=0
                    )
                    Re_cz = np.concatenate(
                        (Re_cz, np.array(file["tasks"]["Re_cz"])), axis=0
                    )
                    Ro_cz = np.concatenate(
                        (Ro_cz, np.array(file["tasks"]["Ro_cz"])), axis=0
                    )
                    gradT_cz = np.concatenate(
                        (gradT_cz, np.array(file["tasks"]["<(grad T)^2>_cz"])), axis=0
                    )
                    gradu_cz = np.concatenate(
                        (gradu_cz, np.array(file["tasks"]["<(grad u)^2>_cz"])), axis=0
                    )
                    qT_cz = np.concatenate(
                        (qT_cz, np.array(file["tasks"]["<QT>_cz"])), axis=0
                    )
                    RfwT_cz = np.concatenate(
                        (RfwT_cz, np.array(file["tasks"]["Ra*<wT>_cz"])), axis=0
                    )
    print("Done.")
    scASI, scAEI = get_ave_indices(sc_time)

    KE_run_ave = rolling_average(KE, sc_time, window=float(args["--window"]))
    Re_run_ave = rolling_average(Re, sc_time, window=float(args["--window"]))

    fig, ax = plt.subplots()
    Re_ax = ax.twinx()
    ax.scatter(sc_time, KE, c='k', marker='+', alpha=0.8)
    Re_ax.scatter(sc_time, Re, c='b', marker='+', alpha=0.8)
    ax.plot(sc_time, KE_run_ave, c='r')
    Re_ax.plot(sc_time, Re_run_ave, c='cyan')
    ax.set_xlabel(r"$\tau_\nu$")
    ax.set_ylabel(r"KE")
    Re_ax.set_ylim([None, 1.5 * np.max(Re[scASI:scAEI])])
    ax.set_ylim([None, 1.5 * np.max(KE[scASI:scAEI])])
    Re_ax.set_ylabel("Re", color="blue")
    Re_ax.tick_params(axis="y", labelcolor="blue")
    ax.axvspan(sc_time[scASI], sc_time[scAEI], color='grey', alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath.joinpath("time_tracks.pdf"))

# #! Flux Profiles
if fluxes or temp_profile or nusselt:
    print("Loading profiles...")
    horiz_files = list(direc.glob("horiz_aves/horiz_aves_s*.h5"))
    horiz_files.sort(key=lambda f: int(re.sub("\D", "", str(f))))
    for i, hfile in enumerate(horiz_files):
        if i == 0:
            with h5.File(hfile, "r") as file:
                horiz_time = np.array(file["scales"]["sim_time"])
                z = np.array(file["tasks"]["<T>"].dims[3]["z"])
                F_cond = np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]
                F_conv = np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]
                Temp = np.array(file["tasks"]["<T>"])[:, 0, 0, :]
        else:
            with h5.File(hfile, "r") as file:
                horiz_time = np.concatenate(
                    (horiz_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                F_cond = np.concatenate(
                    (F_cond, np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]), axis=0
                )
                F_conv = np.concatenate(
                    (F_conv, np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]), axis=0
                )
                Temp = np.concatenate(
                    (Temp, np.array(file["tasks"]["<T>"])[:, 0, 0, :]), axis=0
                )
    hASI, hAEI = get_ave_indices(horiz_time)
    hwidth = float(args["--Hwidth"])
    hl = np.abs(z - hwidth).argmin()
    hu = np.abs(z - (z[-1] - hwidth)).argmin()
    print("Done.")

if fluxes:
    print("Creating Flux Profiles...")
    F_tot = F_cond + F_conv
    F_cond_bar = np.nanmean(F_cond[hASI:hAEI], axis=0)
    F_conv_bar = np.nanmean(F_conv[hASI:hAEI], axis=0)
    F_tot_bar = np.nanmean(F_tot[hASI:hAEI], axis=0)

    heat_func = get_heat_func()
    F_imp = cumulative_trapezoid(heat_func, z, initial=0)

    cz_disc = np.abs((F_imp[hl:hu] - F_tot_bar[hl:hu]) / F_tot_bar[hl:hu])
    discrepency = np.trapz(cz_disc, z[hl:hu])*100
    max_disc = np.max(cz_disc) * 100
    print(f"\tmax discrepency = {max_disc:.2f}%\n\tave discrepence = {discrepency:.1f}%")

    fig, ax = plt.subplots()
    ax.plot(F_cond_bar, z, c='b', label=r"F$_{cond}$")
    ax.plot(F_conv_bar, z, c='r', label=r"F$_{conv}$")
    ax.plot(F_imp, z, c='purple', label=r'F$_{imp}$')
    ax.plot(F_tot_bar, z, c='k', ls='--', label=r"F$_{tot}$")
    ax.set_xlabel("Flux")
    ax.set_ylabel("z")
    ax.axhspan(z[0], z[hl], color='r', alpha=0.2)
    ax.axhspan(z[hu], z[-1], color='b', alpha=0.2)
    plt.legend()
    plt.savefig(outpath.joinpath("flux.pdf"))

#! Temp profile
if temp_profile:
    print("Creating Temp Profile...")
    T_bar = np.nanmean(Temp[hASI:hAEI], axis=0)
    print("Done.")
    Tl_cz = T_bar[hl]
    Tl = T_bar[0]
    Tu_cz = T_bar[hu]
    Tu = T_bar[-1]
    dT_cz = Tl_cz
    fig, ax = plt.subplots()
    ax.plot(T_bar/T_bar[0], z, c='k')
    ax.set_xlabel('<T>/<T(z=0)>')
    ax.set_ylabel('z')
    ax.axhspan(z[0], z[hl], color='r', alpha=0.2)
    ax.axhspan(z[hu], z[-1], color='b', alpha=0.2)
    plt.savefig(outpath.joinpath("temp_profile.pdf"))

# # ! Nusselt Number
if nusselt:
    print("Calculating Nusselt Numbers...")
    F_cond_bar = np.nanmean(F_cond[hASI:hAEI], axis=0)
    F_conv_bar = np.nanmean(F_conv[hASI:hAEI], axis=0)
    nu_CZ = 1 + (
        trapezoid(F_conv_bar[hl:hu], z[hl:hu], axis=0) / trapezoid(F_cond_bar[hl:hu], z[hl:hu], axis=0)
    )
    nu_box = 1 + (
        trapezoid(F_conv_bar, z, axis=0) / trapezoid(F_cond_bar, z, axis=0)
    )

    outvalues['Nu_cz'] = nu_CZ
    outvalues['Nu_box'] = nu_box

    print(f"\tNu(CZ) = {nu_CZ:.2f}\n\tNu(box) = {nu_box:.2f}")

if power_integrals:
    scalar_files = list(direc.glob("scalars/scalars_s*.h5"))
    scalar_files.sort(key=lambda f: int(re.sub("\D", "", str(f))))
    print("Loading scalars...")
    for i, sc_file in enumerate(scalar_files):
        if i == 0:
            with h5.File(sc_file, "r") as file:
                sc_time = np.array(file["scales"]["sim_time"])
                gradT = np.array(file["tasks"]["<(grad T)^2>"])[:, 0, 0, 0]
                gradu = np.array(file["tasks"]["<(grad u)^2>"])[:, 0, 0, 0]
                qT = np.array(file["tasks"]["<QT>"])[:, 0, 0, 0]
                RfwT = np.array(file["tasks"]["Ra*<wT>"])[:, 0, 0, 0]
        else:
            with h5.File(sc_file, 'r') as file:
                sc_time = np.concatenate((sc_time, np.array(file['scales']['sim_time'])))
                gradT = np.concatenate((gradT, np.array(file['tasks']['<(grad T)^2>'])[:, 0, 0, 0]), axis=0)
                gradu = np.concatenate((gradu, np.array(file['tasks']['<(grad u)^2>'])[:, 0, 0, 0]), axis=0)
                qT = np.concatenate((qT, np.array(file['tasks']['<QT>'])[:, 0, 0, 0]), axis=0)
                RfwT = np.concatenate((RfwT, np.array(file['tasks']['Ra*<wT>'])[:, 0, 0, 0]), axis=0)
    scASI, scAEI = get_ave_indices(sc_time)
    gradTbar = np.nanmean(gradT[scASI:scAEI])
    qTbar = np.nanmean(qT[scASI:scAEI])
    temp_disc = np.abs((qTbar-gradTbar)/gradTbar)*100
    gradubar = np.nanmean(gradu[scASI:scAEI])
    RfwTbar = np.nanmean(RfwT[scASI:scAEI])
    visc_disc = np.abs((RfwTbar - gradubar)/gradubar) * 100
    print(f"<gradT> = {gradTbar:.2e}, <QT> = {qTbar:.2e}, disc: {temp_disc:.2f}%")
    print(f"<gradu> = {gradubar:.2e}, Rf*<wT> = {RfwTbar:.2e}, disc: {visc_disc:.2f}%")

    outvalues['gradT'] = gradTbar
    outvalues['QT'] = qTbar
    outvalues['gradu'] = gradubar
    outvalues['RfwT'] = RfwTbar

    Nu_ss = 1 / gradTbar
    outvalues['Nu_ss'] = Nu_ss
    print(f"SS Nu = {Nu_ss:.2f}")

if reynolds:
    print("Reynolds number not yet implemented")

if rossby:
    print("Rossby number not yet implemented")

if args['--gif']:
    print("Load Snapshots...")
    snap_files = sorted(list(direc.glob('snapshots/*.h5')), key=lambda f: int(re.sub("\D", "", f.name)))
    for i, sfile in enumerate(snap_files):
        if i==0:
            with h5.File(sfile, 'r') as f:
                sntime = np.array(f['tasks']['Temp'].dims[0]['sim_time'])
                y = np.array(f['tasks']['Temp'].dims[2]['y'])
                z = np.array(f['tasks']['Temp'].dims[3]['z'])
                Temp = np.array(f['tasks']['Temp'])[:, 0, :, :]
        else:
            with h5.File(sfile, 'r') as f:
                sntime = np.concatenate((sntime, np.array(f['tasks']['Temp'].dims[0]['sim_time'])), axis=0)
                Temp, y, z = combine_array(Temp, f['tasks']['Temp'], y, z, axis=0)
    print("Done")
    snASI, snAEI = get_ave_indices(sntime)
    sntime = sntime[snASI:snAEI]
    Temp = Temp[snASI:snAEI, :, :]
    yy, zz = np.meshgrid(z, y)
    direc.joinpath('plots').mkdir(exist_ok=True)
    fnames = []
    for tidx, t in enumerate(sntime):
        print(f"Plotting Frames... {tidx+1}/{len(sntime)}", end='\r')
        plt.contourf(zz, yy, Temp[tidx], 50, cmap='inferno')
        plt.colorbar(label='T')
        plt.title(fr"$\tau_\nu$ = {t:.2f}")
        plt.xlabel('y')
        plt.ylabel('z')
        fnames.append(direc.joinpath(f"plots/{tidx:0>3}.png"))
        plt.savefig(fnames[-1], dpi=500)
        plt.clf()
    print("\nDone.")
    with iio.get_writer(direc.joinpath('images/Dynamics.gif'), mode='I') as writer:
        for i, fname in enumerate(fnames):
            print(f"Creating Gif... frame {i+1}/{len(fnames)}", end='\r')
            image = iio.imread(fname)
            writer.append_data(image)
    print("\nDone.")

with open(direc.joinpath("outscalars.json"), "w") as f:
    json.dump(
        outvalues,
        f,
        indent=4,
    )
