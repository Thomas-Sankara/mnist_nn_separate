import torch.nn.functional as F


def loss_func():
    return F.cross_entropy


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()  # 反向传播产生的是每个参数对应的梯度
        opt.step()  # 这步会用梯度和学习率更新参数，用这函数会自动避免把更新参数的计算步骤加入计算图
        opt.zero_grad()

    return loss.item(), len(xb)
