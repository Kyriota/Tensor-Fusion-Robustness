import torch
import numpy as np



def InitPert(scale_factor=0.001, device='cuda'):
    pert = torch.randn(1, 1, 28, 28).to(device)
    pert = (pert - pert.min()) / (pert.max() - pert.min())
    pert *= scale_factor
    return pert



def Attack(
        model,
        test_loader,
        upper_bound,
        lower_bound,
        batch_size=4096,
        max_iter=50000,
        lr=0.001,
        device='cuda'
    ):

    def AttackStep(
            x_batch,
            y_batch,
            perts,
            status_indicators,
            step_indicators,
            slot_indicators,
            max_iter
        ):
        # perform one step of attack
        x_adv = x_batch + perts
        y_pred = model(x_adv)

        # check if attack success
        status_indicators[(y_pred.argmax(dim=1) != y_batch.argmax(dim=1)) & (slot_indicators.cuda() == 1)] = 1

        # get loss of target label
        loss = y_pred[status_indicators == 0, y_batch.argmax(dim=1)[status_indicators == 0]].sum()
        loss.backward()

        # normalize pert grad of each sample
        pert_grad_norm = perts.grad.detach().view(perts.shape[0], -1).norm(dim=1)
        pert_grad_norm[pert_grad_norm == 0] = -1 # set zero grad norm to -1 to avoid division by zero
        # update perts that have positive grad norm
        perts.data[pert_grad_norm > 0] = (perts - lr * perts.grad.detach() / pert_grad_norm.view(-1, 1, 1, 1))[pert_grad_norm > 0]
        # x_batch+perts should be within [lower_bound, upper_bound]
        perts.data[x_batch + perts < lower_bound] = lower_bound - x_batch[x_batch + perts < lower_bound]
        perts.data[x_batch + perts > upper_bound] = upper_bound - x_batch[x_batch + perts > upper_bound]
        # mark zero grad norm as attack failed
        status_indicators[(pert_grad_norm.cpu() == -1) & (status_indicators == 0) & (slot_indicators == 1)] = -1

        # reset grad
        perts.grad.zero_()
        # update step
        step_indicators += 1

        # check if attack failed
        status_indicators[step_indicators == max_iter] = -1


    model.eval()

    if batch_size > len(test_loader):
        batch_size = len(test_loader)

    radius_his = torch.ones(len(test_loader)) * -1 # -1 means attack failed, -2 means misclassified
    pert_his = np.array([None] * len(test_loader)) # None means attack failed
    x_batch, y_batch = torch.zeros(batch_size, 1, 28, 28).to(device), torch.zeros(batch_size, 10).to(device)
    slot_indicators = torch.zeros(batch_size) # 0 means empty, 1 means filled
    index_indicators = torch.ones(batch_size) * -1 # record index of data in data loader
    status_indicators = torch.zeros(batch_size) # 0 means attempting attack, 1 means attack success, -1 means attack failed
    step_indicators = torch.zeros(batch_size) # record step of attack
    perts = torch.zeros(batch_size, 1, 28, 28).to(device)
    perts.requires_grad = True

    for i in range(batch_size):
        perts.data[i] = InitPert(device=device)

    last_batch = False

    for i, (x, y) in enumerate(test_loader):
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        x, y = x.to(device), y.to(device)

        # check if current sample is misclassified
        y_pred = model(x)
        if y_pred.argmax(dim=1) != y.argmax(dim=1):
            radius_his[i] = -2
            continue

        slot_i = slot_indicators.tolist().index(0)
        index_indicators[slot_i] = i

        # fill current sample to batch
        x_batch[slot_i] = x
        y_batch[slot_i] = y
        slot_indicators[slot_i] = 1

        if i == len(test_loader) - 1:
            last_batch = True

        # do attack step by step
        while slot_indicators.sum() == batch_size or (last_batch and slot_indicators.sum() > 0):
            AttackStep(x_batch, y_batch, perts, status_indicators, step_indicators, slot_indicators, max_iter)

            # check if attack success or failed
            if status_indicators.abs().sum() != 0:
                loader_indices = index_indicators[status_indicators == 1].int().tolist()
                for i, loader_index in enumerate(loader_indices):
                    pert_his[loader_index] = perts[status_indicators == 1][i].detach().cpu().numpy()
                    radius_his[loader_index] = perts[status_indicators == 1][i].view(784).norm().item()
                    
                    # print attack success info
                    print(' - Index {} \t attack success at iter: {}, \t radius: {:.4f}'.format(
                        loader_index,
                        step_indicators[status_indicators == 1][i].int().item(),
                        radius_his[loader_index]
                    ))
                
                loader_indices = index_indicators[status_indicators == -1].int().tolist()
                for i, loader_index in enumerate(loader_indices):
                    # print attack failed info
                    print(' - Index {} \t attack failed at iter: {}'.format(
                        loader_index,
                        step_indicators[status_indicators == -1][i].int().item()
                    ))

                # reset slots that attack success or failed
                slot_indicators[status_indicators != 0] = 0
                index_indicators[status_indicators != 0] = -1
                step_indicators[status_indicators != 0] = 0
                batch_indices = torch.arange(batch_size)[status_indicators != 0]
                for batch_index in batch_indices:
                    perts.data[batch_index] = InitPert(device=device)
                    x_batch[batch_index].data.zero_()
                    y_batch[batch_index].data.zero_()

                status_indicators[status_indicators != 0] = 0

                if not last_batch:
                    break
    
    return radius_his.tolist(), pert_his