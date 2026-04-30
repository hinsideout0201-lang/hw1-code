import numpy as np

from model import MLP
from loss import CrossEntropyLoss
from train import train, load_weights, accuracy
from data import load_eurosat
import matplotlib.pyplot as plt
from error import show_error

X, y = load_eurosat("./EuroSAT_RGB")

n = X.shape[0]
idx = np.random.permutation(n)

train_end = int(0.7*n)
val_end = int(0.85*n)

train_idx = idx[:train_end]
val_idx = idx[train_end:val_end]
test_idx = idx[val_end:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]


input_dim = X.shape[1]
hidden_dim = 128
num_classes = 10

model = MLP(input_dim,hidden_dim,num_classes,activate='ReLU')
loss = CrossEntropyLoss()

best_weights, train_losses, val_losses, val_accs = train(model, loss,
                                                         X_train, y_train,
                                                         X_val, y_val,
                                                         epochs=10,lr=0.05,
                                                         batch_size=256)

load_weights(model,best_weights)
np.save("best_weights.npy",np.array(best_weights,dtype=object))
val_acc = accuracy(model,X_val,y_val)
print("final val acc:",val_acc)

def confusion_matrix(model, X, y, num_classes):
    output = model.forward(X)
    pred = np.argmax(output,axis=1)
    cm = np.zeros((num_classes,num_classes),dtype=int)
    for i in range(len(y)):
        cm[y[i], pred[i]] += 1

    return cm

test_acc = accuracy(model,X_test,y_test)
print("test acc:",test_acc)

cm = confusion_matrix(model,X_test,y_test,num_classes)
print("confusion matrix:")
print(cm)

show_error(model, X_test, y_test, num_classes)

plt.plot(train_losses,label="train loss")
plt.plot(val_losses,label="val loss")
plt.legend()
plt.title("loss")
plt.savefig("loss.png")
plt.close()

plt.plot(val_accs,label="val acc")
plt.legend()
plt.title("accuracy")
plt.savefig("accuracy.png")
plt.close()


first_layer = None
for layer in model.layers:
    if hasattr(layer, "W"):
        first_layer = layer
        break

W = first_layer.W
num_show = 8

plt.figure(figsize=(10, 5))

for i in range(num_show):
    w = W[:, i]
    img = w.reshape(64, 64, 3)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Neuron {i}")

plt.suptitle("First Layer Weights Visualization")
plt.savefig("visualization.png")
plt.close()


learning_rates = [0.01, 0.001]
hidden_dims = [64, 128]
weight_decays = [1e-4]
results = []

for lr in learning_rates:
    for hd in hidden_dims:
        for wd in weight_decays:

            print(f"=== lr={lr}, hidden={hd}, wd={wd} ===")
            model = MLP(input_dim,hd,num_classes,activate='ReLU')
            loss = CrossEntropyLoss()

            best_weights, train_losses, val_losses, val_accs = train(
                model,loss,X_train,y_train,X_val,y_val,
                epochs=10,lr=lr,batch_size=256,weight_decay=wd)

            load_weights(model,best_weights)
            val_acc = accuracy(model,X_val,y_val)
            print("val acc:", val_acc)
            results.append((lr, hd, wd, val_acc))

best = max(results, key=lambda x: x[3])
print("best result:", best)

print("\n=== All Results ===")
for r in results:
    print(f"lr={r[0]}, hidden={r[1]}, wd={r[2]} -> acc={r[3]:.4f}")
