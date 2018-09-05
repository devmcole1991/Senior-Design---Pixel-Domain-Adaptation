# Open up in Spyder for better visualization.
import pickle

# Getting losses
with open('losses.pkl', 'rb') as f:  
    d1_loss_l, d1_accuracy_l, d2_loss_l, d2_accuracy_l, g0, g1, g2, g3, g4 = pickle.load(f)
    
# Manipulate data here!