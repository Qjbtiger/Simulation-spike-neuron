from functools import partial, wraps
from line_profiler import LineProfiler
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
def LIF(N, epsilon, ratio, J0, g, Vs, Vr, tau, tauReset, nu, delta, T, dt):
    '''
    Parameters:
        N: The number of neurons
        epsilon: The ratio of Average number of connections per excitatory neurons
        ratio: EI ratio
        J0: The standrad connection strength
        Vs: The spike threshold voltage
        Vr: The reset voltage
        tau: The membrane time
        delta: The delay time
        T: Total simulation time
        dt: Inteval of simulation time
    '''

    Ne = int(ratio * N)
    Ni = N - Ne

    Ce = int(epsilon * Ne)
    gamma = Ni / Ne
    Ci = int(gamma * Ce)
    C = Ce + Ci
    mask = np.zeros(shape=(N, N))
    mask[np.random.rand(N, N) < epsilon] = 1

    numIntervals = int(T // dt)
    deltaIntervals = int(delta // dt)

    # connection stength
    Jee, Jei = J0, J0
    Jie, Jii = g*J0, g*J0
    Jext = J0
    J = np.zeros(shape=(N, N))
    J[:Ne, :Ne] = Jee
    J[:Ne, Ne:] = -Jei
    J[Ne:, :Ne] = Jie
    J[Ne:, Ne:] = -Jii
    J *= mask
    if epsilon<=0.1:
        # sparse to increase speed
        J = sp.csr_matrix(J)
    nuThr = Vs * 1000 / (J0 * Ce * tau)
    nuExt = nu * nuThr
    externalSpike = np.random.poisson(dt * nuExt * Ce / 1000, size=(numIntervals, N))
    
    if deltaIntervals == 0:
        # At least store the last iteration result for simualtion
        deltaIntervals = 1
    spikeTimes = []
    spikeIndexes = []
    # store current voltage and synaptic current
    voltage = np.random.rand(N) * (Vs - Vr) + Vr
    # store voltage before delta step
    spikeBeforeDelta = np.zeros(shape=(deltaIntervals, N))
    # store the refractory period neurons
    refractoryNeurons = np.zeros(shape=N)
    indexesList = np.arange(N)
    indexOfDeltaLists = 0
    for i in range(numIntervals):
        t = (i+1) * dt

        # check spike
        refractoryNeurons -= dt
        spikeOrNot = (voltage >= Vs)
        spikeBeforeDelta[indexOfDeltaLists] = spikeOrNot.astype(np.int64)
        numSpike = np.sum(spikeOrNot)
        if numSpike > 0:
            refractoryNeurons[spikeOrNot] = np.repeat(tauReset, repeats=numSpike)
            voltage[spikeOrNot] = np.repeat(Vr, repeats=numSpike)
            spikeTimes.extend([t] * numSpike)
            spikeIndexes.extend(list(indexesList[spikeOrNot]))

        # update voltage
        indexOfDeltaLists = (indexOfDeltaLists + 1) % deltaIntervals
        current = tau * J.dot(spikeBeforeDelta[indexOfDeltaLists])

        voltage = np.where(refractoryNeurons <= 0, voltage + dt * (- voltage + current) / tau + externalSpike[i] * Jext, voltage)

    return np.asarray(spikeTimes), np.asarray(spikeIndexes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=12500, help="The number of neurons")
    parser.add_argument("--epsilon", type=float, default=0.1, help="The ratio of average number of connection per excitatory neurons Ce/Ne")
    parser.add_argument("-r", "--ratio", type=float, default=0.8, help="EI ratio")
    parser.add_argument("-J", type=float, default=0.1, help="The standrad connection strength")
    parser.add_argument("-g", type=float, default=6.0, help="The ratio of Ji/Je")
    parser.add_argument("--Vs", type=float, default=20.0, help="The spike threshold voltage")
    parser.add_argument("--Vr", type=float, default=10.0, help="The reset voltage")
    parser.add_argument("--tau", type=float, default=20.0, help="The membrane time, \\tau_E = \\tau_I")
    parser.add_argument("--tau_reset", type=float, default=2.0, help="The refractory period time")
    parser.add_argument("--nu", type=float, default=4.0, help="Possion process of external stimula")
    parser.add_argument("--delta", type=float, default=1.5, help="The delay time (ms)")
    parser.add_argument("-T", type=float, default=100, help="Total simulation time (ms)")
    parser.add_argument("--dt", type=float, default=0.1, help="The interval time in simulation")
    args = parser.parse_args()

    N = args.N
    epsilon = args.epsilon
    ratio = args.ratio
    J = args.J
    g = args.g
    Vs = args.Vs
    Vr = args.Vr
    tau = args.tau
    tauReset = args.tau_reset
    nu = args.nu
    delta = args.delta
    T = args.T
    dt = args.dt

    def calc():
        start = time.time()
        spikeTimes, spikeIndexes = LIF(N, epsilon, ratio, J, g, Vs, Vr, tau, tauReset, nu, delta, T, dt)
        print("Cost time: {}".format(time.time() - start))

        return spikeTimes, spikeIndexes

    spikeTimes, spikeIndexes = calc()

    def plot(spikeAx, histAx, spikeTimes, spikeIndexes):
        randomChosenIndexes = spikeIndexes < 100
        spikeTimesSmall = spikeTimes[randomChosenIndexes]
        spikeIndexesSmall = spikeIndexes[randomChosenIndexes]

        spikeAx.clear()
        histAx.clear()
        spikeAx.plot(spikeTimesSmall, spikeIndexesSmall, '.', markersize=1.0)
        spikeAx.set_xticks([])
        spikeAx.set_ylabel("index")
        spikeAx.legend()
        histAx.hist(spikeTimes, bins=int(T // dt))
        histAx.set_xlabel("time (ms)")
        fig.canvas.draw_idle()

    fig = plt.figure()
    grid = plt.GridSpec(2, 1)
    spikeAx = plt.subplot(grid[0, :])
    histAx = plt.subplot(grid[1, :])
    plot(spikeAx, histAx, spikeTimes, spikeIndexes)

    # argsList = [g, nu]
    # labelsList = ["g", "$\\nu$"]
    # valMinList = [0.1, 0.5]
    # valMaxList = [8, 6]
    # valStepList = [0.1, 0.1]
    # sliderList = []
    # def update(i, val):
    #     argsList[i] = val
    # for i in range(len(argsList)):
    #     sliderAx = plt.subplot(grid[10 + 2*i, :])
    #     sliderList.append(Slider(sliderAx, labelsList[i], valMinList[i], valMaxList[i], argsList[i], valstep=valStepList[i]))
    #     sliderList[-1].on_changed(partial(update, i))
    # def resetButtonOnClick(mouse_event):
    #     spikeTimes, spikeIndexes = calc()
    #     plot(spikeAx, histAx, spikeTimes, spikeIndexes)
    # buttionAx = plt.subplot(grid[10 - 1 + 2*len(argsList), -4:-2])
    # resetButton = Button(buttionAx, "Re-calc")
    # resetButton.on_clicked(resetButtonOnClick)
    # plt.savefig("./N{}-epsilon{}-r{}-J{}-g{}-Vs{}-Vr{}-tau{}-tauReset{}-nu{}-delta{}-T{}.png".format(N, epsilon, ratio, J, g, Vs, Vr, tau, tauReset, nu, delta, T))
    plt.show()
    plt.close()
        






