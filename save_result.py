import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


def save_result(trainer):
    typeOfDataToMetricsObject = trainer.typeOfDataToMetricsObject
    SetOfMatrics = set()
    for metrics in typeOfDataToMetricsObject.values():
        for Metrics_function in metrics.Metrics_functions:
            SetOfMatrics.add(Metrics_function)
    path = "results/"+datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    for Metrics_function in SetOfMatrics:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for name, metrics in typeOfDataToMetricsObject.items():
            if Metrics_function in metrics.Metrics_functions:
                the_metrics_function_result = metrics.Metrics_functions_to_Metrics_result.get(Metrics_function)
                ax.plot(np.arange(0, len(the_metrics_function_result)),
                        the_metrics_function_result,
                        label=Metrics_function.__name__ + " on the " + name)
        ax.grid(True)
        ax.set_xlabel('epoch')
        ax.set_title(Metrics_function.__name__ + "\n" + trainer.Optimizer.name + ':' + trainer.Optimizer.info + "\n"
                     + trainer.Model.name + ": " + trainer.Model.info + "\n" + trainer.info)
        plt.legend(loc="best")
        plt.savefig(path+'/'+Metrics_function.__name__+datetime.now().strftime("%d.%m.%Y-%H.%M.%S")+'.png')
        plt.show()

