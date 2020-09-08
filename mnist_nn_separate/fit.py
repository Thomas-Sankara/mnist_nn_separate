import numpy as np
import loss_functions
import torch


'''
请注意，我们总是在训练之前调用model.train()，并在推断之前调用model.eval()，
因为诸如nn.BatchNorm2d和nn.Dropout之类的图层会使用它们，以确保这些不同阶段的行为正确。

model.train()
启用 BatchNormalization 和 Dropout
model.eval()
不启用 BatchNormalization 和 Dropout

bn层和dropout是两种网络训练中用到的处理技巧，因为很有效，就被整合到一般性的网络运行步骤中了。
从函数名你也能看出来，model.train()是训练时用的，model.eval()是验证(evaluate)时用的。
'''


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # 训练，启用 BatchNormalization 和 Dropout。这两个本来只是后提出来的用来提升网络训练效果的算法组件，
        # 但是由于太好使太通用了，就被整合进网络中。但是这俩组件是在训练时使用的，所以训练前要显式打开，验证前要显式关闭。
        model.train()
        for xb, yb in train_dl:
            loss_functions.loss_batch(model, loss_func, xb, yb, opt)

        # 验证，不启用 BatchNormalization 和 Dropout
        model.eval()
        with torch.no_grad():
            '''
            先讲讲python语言特性——[a(b) for b in c]：
            如果先不看外面的list强转符号，里面就是：
            1、先用b从c里迭代取值，注意返回的是个迭代器，不是完成取值组成list或元组了，是迭代器
            2、把这个迭代器结果输入到函数a()中得到结果，这个结果也是个迭代器，所以你会发现，如果没有外面的[]符号，报错
            最后，外面的list强转符号可以把迭代器得到的所有结果整合进一个list中
            '''
            '''
            zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回zip对象。
            如果各个迭代器的元素个数不一致，则返回zip对象里的长度与最短的输入相同。
            利用*号操作符，即zip(*)可以将zip对象、元组、list解压为元组，有需要可以自己强转list。
            所以你发现，print一下losses和nums，他们是元组。
            '''
            losses, nums = zip(
                *[loss_functions.loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)
