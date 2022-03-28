
#James Oswald 1/12/21

import numpy as np
import matplotlib.pyplot as plt
import argparse

#takes a file path and returns a matplot figure object
def genFigure(filePath):
    data = np.load(filePath)
    alpha, beta, gamma, delta, latentDims, lstmLayers, drop, lstmInfo, linearWidth = data["par"]

    trainMat = data["tl"]
    epochsT = trainMat[:,0]
    rLossT = trainMat[:,1]
    kdLossT = trainMat[:,2]
    regLossT = trainMat[:,3]
    lossT = trainMat[:,4]
    accT = trainMat[:,5]

    #validation losses vs epoch
    validMat = data["vl"]
    epochsV = validMat[:,0]
    rLossV = validMat[:,1]
    kdLossV = validMat[:,2]
    regLossV = validMat[:,3]
    lossV = validMat[:,4]
    accV = validMat[:,5]

    #corelations with each dimention vs epoch
    wldimCors = data["wldims"]
    lidimCors = data["liidims"]
    dimRange = [j * 50 for j in range(wldimCors.shape[0])]

    #fig, ax = plt.subplots(3)
    fig, ax = plt.subplots(5)
    fig.set_size_inches(8, 7)
    fig.suptitle(r"$\alpha$=" + str(alpha) + r" $\beta$=" + str(beta) + r" $\gamma$=" + str(gamma) + r" $\delta$=" + str(delta) + " latentDims=" + str(latentDims) + " lstmLayers=" + str(lstmLayers) + " Dropout=" + str(drop) + " lstmInfo=" + str(lstmInfo) + " linearWidth" + str(linearWidth), fontsize=12)

    ax[0].plot(epochsT, rLossT, label="r Loss")
    ax[0].plot(epochsT, kdLossT, label="kld Loss")
    ax[0].plot(epochsT, regLossT, label="reg Loss")
    ax[0].plot(epochsT, lossT, label="Training Loss")
    ax[0].plot(epochsV, lossV, label="Valid Loss", ls="--")
    ax[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[0].set(xlabel="epoch", ylabel="Training Loss")
    ax[0].label_outer()

    ax[1].plot(epochsT, rLossV, label="r Loss")
    ax[1].plot(epochsT, kdLossV, label="kld Loss")
    ax[1].plot(epochsT, regLossV, label="reg Loss")
    ax[1].plot(epochsV, lossV, label="Valid Loss")
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].set(xlabel="epoch", ylabel="Validation Loss")
    ax[1].label_outer()

    ax[2].plot(epochsT, accT, label="Train Accuracy")
    ax[2].plot(epochsV, accV, label="Valid Accuracy")
    ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax[2].set(xlabel="epoch",ylabel="Accuracy")
    ax[2].text(x=1300, y=0.62, s=f"Training Accuracy: {'{:.3f}'.format(accT[len(accT) - 1])}")
    ax[2].text(x=1300, y=0.50, s=f"Testing Accuracy: {'{:.3f}'.format(accV[len(accV) - 1])}")
    ax[2].label_outer()

    for i in range(int(latentDims)):
        ax[3].plot(dimRange, wldimCors[:,i], label=str(i), ls="-" if i == 0 else "--")
        ax[4].plot(dimRange, lidimCors[:,i], label=str(i), ls="-" if i == 1 else "--")
    ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2)
    ax[3].set(xlabel="epoch",ylabel="WL Cor")
    ax[3].label_outer()
    ax[4].legend(loc="center left", bbox_to_anchor=(1, 0.3), ncol=2)
    ax[4].set(xlabel="epoch",ylabel="LII Cor")
    ax[4].label_outer()

    fig.subplots_adjust(right=0.75)
    
    return fig

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="i", default="None")
parser.add_argument("-o", "--output", dest="o", default="None")
args = parser.parse_args()

""" if args.i != "None":
    f = genFigure(args.i)
    name = ""
    if args.o == "None":
        raise Exception("No output name provided")
        #name = "./graphs/b"+args.i+".png"
    else:
        name = args.o
    f.savefig(name)
else:
    fileName = "./runs/1623311492.npz"
    f = genFigure(fileName)
    f.savefig("./graphs/latest.png") """

def genPlotForRun(runsPath, run, graphsPath, graph):
    f = genFigure(runsPath+"/"+run)
    f.savefig(graphsPath+"/"+graph)


#genFigure("all-results/1-18-22-res/runs/a19lds19b0.007g1.0d1.0h13.npz")