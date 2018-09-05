# Open up in Spyder for better visualization.
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Getting losses
with open('losses.pkl', 'rb') as f:  
    d1_loss_l, d1_accuracy_l, d2_loss_l, d2_accuracy_l, g0, g1, g2, g3, g4 = pickle.load(f)
    
# Manipulate data here!
def make_graph(value, model, number, metric):
    x = np.array(range(0,30000))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(x, value)
    plt.xlabel('Iterations')
    plt.ylabel(model + ' ' + str(number) + ' ' + metric)
    plt.title(model + ' ' + str(number) + ' ' + metric)
    fig.savefig(model + ' ' + str(number) + ' ' + metric)

make_graph(d1_loss_l, 'Discriminator', 1, 'Loss')
make_graph(d1_accuracy_l, 'Discriminator', 1, 'Accuracy')  
print(sum(d1_accuracy_l)/30000)
make_graph(d2_loss_l, 'Discriminator', 2, 'Loss')
make_graph(d2_accuracy_l, 'Discriminator', 2, 'Accuracy')
print(sum(d2_accuracy_l)/30000)
make_graph(g0, 'Generator', 1, 'Loss')
make_graph(g1, 'Generator', 2, 'Loss')