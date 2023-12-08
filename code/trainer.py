import torch



def Validate(model, val_loader, criterion, device='cuda'):
    model.eval()

    loss_his, acc_his = [], []

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            # one-hot to label
            y = torch.nn.functional.one_hot(y, num_classes=10).float()

            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss_his.append(loss.item())
            acc_his.append((y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean().item())

    return loss_his, acc_his


def Train(
        model,
        train_loader,
        epochs=1,
        lr=1e-3,
        weight_decay=0,
        device='cuda',
        val_freq=0.5, # non-zero means use validation
        val_loader=None,
        patience=10 # early stopping
    ):
    # set optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    loss_his_train, acc_his_train = [], []
    loss_his_train_all, acc_his_train_all = [], []
    loss_his_val, acc_his_val = [], []

    val_step = int(len(train_loader) * val_freq)

    step_counter = 0
    patience_counter = 0

    best_params = None # saved best params

    # train
    for epoch in range(epochs):
        if patience_counter >= patience:
            print('Early stopping!')
            break
        loss_his_epoch, acc_his_epoch = [], []
        for i, (x, y) in enumerate(train_loader):
            model.train()

            # one-hot to label
            y = torch.nn.functional.one_hot(y, num_classes=10).float()

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            loss_his_epoch.append(loss.item())
            acc_his_epoch.append((y_pred.argmax(dim=1) == y.argmax(dim=1)).float().mean().item())

            if val_freq > 0 and step_counter % val_step == 0 and step_counter > 0:
                loss_his_val_epoch, acc_his_val_epoch = Validate(model, val_loader, criterion, device=device)
                loss_his_val.append(sum(loss_his_val_epoch) / len(loss_his_val_epoch))
                acc_his_val.append(sum(acc_his_val_epoch) / len(acc_his_val_epoch))

                loss_his_train.append(sum(loss_his_epoch) / len(loss_his_epoch))
                acc_his_train.append(sum(acc_his_epoch) / len(acc_his_epoch))

                loss_his_train_all += loss_his_epoch
                acc_his_train_all += acc_his_epoch

                loss_his_epoch, acc_his_epoch = [], []

                print('Epoch: {}, Step: {}, Loss: {:.4f}, Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
                    epoch,
                    step_counter,
                    loss_his_train[-1],
                    acc_his_train[-1],
                    loss_his_val[-1],
                    acc_his_val[-1]
                ))

                # save best params
                if best_params is None or loss_his_val[-1] == min(loss_his_val):
                    print(' - Found better model!')
                    best_params = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            step_counter += 1

    # load best params
    model.load_state_dict(best_params)

    return model, {
        'loss_his_train': loss_his_train,
        'acc_his_train': acc_his_train,
        'loss_his_train_all': loss_his_train_all,
        'acc_his_train_all': acc_his_train_all,
        'loss_his_val': loss_his_val,
        'acc_his_val': acc_his_val,
    }