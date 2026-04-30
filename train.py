import numpy as np

def train(model,loss,X_train,y_train,X_val,y_val,epochs=10,lr=0.01,
          batch_size=256,weight_decay=1e-4,lr_decay=0.95):
    n = X_train.shape[0]
    best_acc = 0
    best_weights = None
    train_losses = []
    val_losses = []
    val_accs = []
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_train = X_train[idx]
        y_train = y_train[idx]

        total_loss = 0

        for i in range(0,n,batch_size):
            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]
            output = model.forward(x)

            cur_loss = loss.forward(output,y)
            l2 = 0
            for layer in model.layers:
                if hasattr(layer,"W"):
                    l2 += np.sum(layer.W**2)
            cur_loss += weight_decay * l2
            
            total_loss += cur_loss
            dx = loss.backward()
            model.backward(dx)
            for layer in model.layers:
                if hasattr(layer, "W"):
                    layer.dW += 2*weight_decay*layer.W
                    layer.W -= lr * layer.dW
                    layer.b -= lr * layer.db
                    
        cur_acc = accuracy(model, X_val, y_val)
        train_losses.append(total_loss)
        val_loss = loss.forward(model.forward(X_val), y_val)
        val_losses.append(val_loss)
        val_accs.append(cur_acc)

        print(f"epoch {epoch+1}, loss {total_loss:.4f}, val acc {cur_acc:.4f}")
        lr *= lr_decay

        if cur_acc > best_acc:
            best_acc = cur_acc
            best_weights = []
            for layer in model.layers:
                if hasattr(layer,"W"):
                    best_weights.append((layer.W.copy(),layer.b.copy()))
    return best_weights, train_losses, val_losses, val_accs

def accuracy(model,X,y):
    output = model.forward(X)
    pred = np.argmax(output, axis=1)
    accuracy = np.mean(pred==y)
    return accuracy

def load_weights(model,best_weights):
    idx = 0
    for layer in model.layers:
        if hasattr(layer,"W"):
            layer.W,layer.b = best_weights[idx]
            idx += 1
