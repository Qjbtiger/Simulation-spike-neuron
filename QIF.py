from functools import partial, wraps
import re
# from line_profiler import LineProfiler
import numpy as np
import scipy.sparse as sp
import time
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider, Button

# 查询接口中每行代码执行的时间
def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs) 
        lp.print_stats() 
        return func_return 
    return decorator

# @nb.njit()
# @func_line_time
def QIF(N, K, Ne, Ni, jee, jei, jie, jii, j0e, j0i, Vs, Vr, tau, delta, T, dt):
    '''
    Parameters:
        N: The number of neurons
        K: Average number of connections per neurons
        Ne: The number of excited neurons
        Ni: The number of inhibited neurons
        jee, jei, jie, jii: The non-normalized connection stength
        j0e, j0i: The non-normalized exteral strenth
        Vs: The spike threshold voltage
        Vr: The reset voltage
        tau: The membrane time
        delta: The delay time
        T: Total simulation time
        dt: Inteval of simulation time
    '''

    # check whether Ne + Ni = N
    if not Ne + Ni == N:
        raise ValueError("Ni + Ne not equal N.")

    p = K / N # probability of connections
    maskEE, maskEI, maskIE, maskII = [np.zeros(shape=(n1, n2)) for n1 in [Ne, Ni] for n2 in [Ne, Ni]]
    for mask in [maskEE, maskEI, maskIE, maskII]:
        mask[np.random.rand(mask.shape[0], mask.shape[1]) < p] = 1
        if p<=0.1:
            # sparse to increase speed
            mask = sp.csr_matrix(mask)

    # covarance stength
    Jee, Jei, Jie, Jii = [i / np.sqrt(K) for i in [jee, jei, jie, jii]]
    J0e, J0i = [i * np.sqrt(K) for i in [j0e, j0i]]

    numIntervals = int(T // dt)
    deltaIntervals = int(delta // dt)
    if deltaIntervals == 0:
        # At least store the last iteration result for simualtion
        deltaIntervals = 1
    spikeTimesE = []
    spikeIndexesE = []
    spikeTimesI = []
    spikeIndexesI = []
    # store current voltage and synaptic current
    voltageE, voltageI = [np.random.rand(n) for n in [Ne, Ni]]
    currentE, currentI = [np.zeros(shape=n) for n in [Ne, Ni]]
    # store voltage before delta step
    spikeBeforeDeltaE, spikeBeforeDeltaI = [np.zeros(shape=(deltaIntervals, n)) for n in [Ne, Ni]]

    indexListE = np.arange(Ne)
    indexListI = np.arange(Ni)
    indexOfDeltaLists = 0
    for i in range(numIntervals):
        t = (i+1) * dt

        # check spike
        def checkSpike(voltage, spikeTimes, spikeIndexes, spikeBeforeDelta, indexLists):
            spikeOrNot = (voltage >= Vs)
            spikeBeforeDelta[indexOfDeltaLists] = spikeOrNot.astype(np.int64)
            numSpike = np.sum(spikeOrNot)
            if numSpike > 0:
                voltage[spikeOrNot] = np.repeat(Vr, repeats=numSpike)
                spikeTimes.extend([t] * numSpike)
                spikeIndexes.extend(list(indexLists[spikeOrNot]))
        checkSpike(voltageE, spikeTimesE, spikeIndexesE, spikeBeforeDeltaE, indexListE)
        checkSpike(voltageI, spikeTimesI, spikeIndexesI, spikeBeforeDeltaI, indexListI)
            

        

        # update voltage
        indexOfDeltaLists = (indexOfDeltaLists + 1) % deltaIntervals
        currentE = tau * (Jee * maskEE.dot(spikeBeforeDeltaE[indexOfDeltaLists]) - Jei * maskEI.dot(spikeBeforeDeltaI[indexOfDeltaLists])) + J0e
        currentI = tau * (Jie * maskIE.dot(spikeBeforeDeltaE[indexOfDeltaLists]) - Jii * maskII.dot(spikeBeforeDeltaI[indexOfDeltaLists])) + J0i
        voltageE += dt * (voltageE * voltageE + currentE) / tau
        voltageI += dt * (voltageI * voltageI + currentI) / tau

    return spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=12500, help="The number of neurons")
    parser.add_argument("-K", type=int, default=100, help="The average number of connection per neuron")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, help="EI ratio")
    parser.add_argument("--jee", type=float, default=1.0, help="The non-normalize connection strength between E-E neorons")
    parser.add_argument("--jei", type=float, default=3.0, help="The non-normalize connection strength between E-I neorons")
    parser.add_argument("--jie", type=float, default=2.0, help="The non-normalize connection strength between I-E neorons")
    parser.add_argument("--jii", type=float, default=2.5, help="The non-normalize connection strength between I-I neorons")
    parser.add_argument("--j0e", type=float, default=1.2, help="The non-normalized external strength for E neorons")
    parser.add_argument("--j0i", type=float, default=0.7, help="The non-normalized external strength for I neorons")
    parser.add_argument("--Vs", type=float, default=1.0, help="The spike threshold voltage")
    parser.add_argument("--Vr", type=float, default=0.0, help="The reset voltage")
    parser.add_argument("--tau", type=float, default=15.0, help="The membrane time")
    parser.add_argument("--delta", type=float, default=0.1, help="The delay time (ms)")
    parser.add_argument("-T", type=float, default=10, help="Total simulation time (ms)")
    parser.add_argument("--dt", type=float, default=0.01, help="The interval time in simulation")
    args = parser.parse_args()

    N = args.N
    K = args.K
    ratio = args.ratio
    jee, jei, jie, jii = args.jee, args.jei, args.jie, args.jii
    j0e = args.j0e
    j0i = args.j0i
    Vs = args.Vs
    Vr = args.Vr
    tau = args.tau
    delta = args.delta
    T = args.T
    dt = args.dt

    def calc():
        start = time.time()
        spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI = QIF(N, K, int(N * ratio), N - int(N * ratio), jee, jei, jie, jii, j0e, j0i, Vs, Vr, tau, delta, T, dt)
        print("Cost time: {}".format(time.time() - start))

        return spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI

    spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI = calc()

    def plot(spikeEAx, histEAxs, spikeIAx, histIAxs, spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI):
        spikeEAx.clear()
        histEAxs.clear()
        spikeIAx.clear()
        histIAxs.clear()
        spikeEAx.plot(spikeTimesE, spikeIndexesE, '.', label="Excited", markersize=1.0)
        spikeEAx.set_xlabel("time (ms)")
        spikeEAx.set_ylabel("index")
        spikeEAx.legend()
        histEAxs.hist(spikeTimesE, bins=1000)
        spikeIAx.plot(spikeTimesI, spikeIndexesI, '.', label="Inhibited", markersize=1.0)
        spikeIAx.set_xlabel("time (ms)")
        spikeIAx.set_ylabel("index")
        spikeIAx.legend()
        histIAxs.hist(spikeTimesI, bins=1000)
        fig.canvas.draw_idle()

    fig = plt.figure()
    grid = plt.GridSpec(16, 4)
    spikeEAx = plt.subplot(grid[0:2, :])
    histEAxs = plt.subplot(grid[4, :])
    spikeIAx = plt.subplot(grid[5:7, :])
    histIAxs = plt.subplot(grid[8, :])
    plot(spikeEAx, histEAxs, spikeIAx, histIAxs, spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI)

    argsList = [N, K]
    labelsList = ["N", "K"]
    valMinList = [500, 50]
    valMaxList = [1500, N]
    valStepList = [100, 50]
    sliderList = []
    def update(i, val):
        argsList[i] = val
    for i in range(len(argsList)):
        sliderAx = plt.subplot(grid[12 + 2*i, :])
        sliderList.append(Slider(sliderAx, labelsList[i], valMinList[i], valMaxList[i], argsList[i], valstep=valStepList[i]))
        sliderList[-1].on_changed(partial(update, i))
    def resetButtonOnClick(mouse_event):
        spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI = calc()
        plot(spikeEAx, histEAxs, spikeIAx, histIAxs, spikeTimesE, spikeTimesI, spikeIndexesE, spikeIndexesI)
    buttionAx = plt.subplot(grid[11 + 2*len(argsList), 2])
    resetButton = Button(buttionAx, "reset")
    resetButton.on_clicked(resetButtonOnClick)
    plt.show()
    plt.close()
        






