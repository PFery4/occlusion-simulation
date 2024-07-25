import matplotlib.pyplot as plt
from src.data.config import COORD_CONV


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.scatter(COORD_CONV['m'], COORD_CONV['px'], marker='x')
    ax.set_xlabel('[m]')
    ax.set_ylabel('[px]')
    ax.set_title('SDD: [px] vs. [m]')
    plt.show()
