import numpy as np
import matplotlib.pyplot as plt

def show_error(model,X_test,y_test,num_classes,num_show=8):
    output = model.forward(X_test)
    pred = np.argmax(output,axis=1)

    wrong_idx = np.where(pred!=y_test)[0]
    print("number of wrong samples:",len(wrong_idx))

    plt.figure(figsize=(10,5))

    for i in range(min(num_show,len(wrong_idx))):
        idx = wrong_idx[i]
        img = X_test[idx].reshape(64, 64, 3)

        plt.subplot(2,4,i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"true:{y_test[idx]},pred:{pred[idx]}")

    plt.suptitle("Misclassified Samples")
    plt.savefig('error.png')
    plt.close()
