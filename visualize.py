import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

file = open("./hist_loss_adv.p",'rb')
hist_p = pickle.load(file)
file.close()

epochs = range(1, len(hist_p["train_pxl"]) + 1)

plt.plot(epochs, hist_p["train_D"], label='train_D')
plt.plot(epochs, hist_p["val_D"], label='val_D')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()
