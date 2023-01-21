import datetime

import matplotlib.pyplot as plt


class Timer():
    def __init__(self) -> None:
        self.timerstarts = datetime.datetime.now()
        self.total_duration = datetime.datetime.now() - datetime.datetime.now()

    def clock_in(self) -> None:
        self.timerstarted = datetime.datetime.now()

    def clock_out(self, printing: bool = False) -> None:
        self.timerpaused = datetime.datetime.now()
        self.total_duration += self.timerpaused - self.timerstarted
        if printing:
            self.check_time()

    def check_time(self) -> None:
        print(self.total_duration.total_seconds())


def plot_dominating_points(X, Y, dominating_points, figsize=[5, 5], save=False, filename='Dominating_points.pdf'):
    plt.figure(figsize=figsize)
    plt.title('Dominating points')
    #plt.contourf(x1s, x2s, yhats.reshape(x1s.shape) >= 0, cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='coolwarm', s=1, alpha=0.3)
    plt.scatter(dominating_points[:, 0], dominating_points[:, 1], c='k', s=10)
    if save: 
        plt.savefig(filename, format='pdf', dpi=100)
        
    plt.show()
