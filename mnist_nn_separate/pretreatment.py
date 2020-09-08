from pathlib import Path
import torch
import pickle
import gzip

# 指定目标路径
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
FILENAME = "mnist.pkl.gz"

def result(bs):
    # 该数据集为 numpy 数组格式，并已使用 pickle(一种用于序列化数据的 python 特定格式）存储。
    # 要强制使用正斜杠表示字符串，请使用as_posix（）方法。
    # "rb"——read binary，以读的方式打开二进制存储格式文件。读numpy array时，参数encoding的值要为"latin-1"。
    # .as_posix()方法会把PosixPath对象变成字符串
    # 可以print一下type，x_train和另外三个变量取出来的时候是numpy.ndarray数据类型
    """
    “Pickling” 是将 Python 对象及其所拥有的层次结构转化为一个字节流的过程，而 “unpickling” 是相反的操作，
    会将（来自一个 binary file 或者 bytes-like object 的）字节流转化回一个对象层次结构。
    上面这段描述告诉你，这个序列化的文件是有结构信息的，你得知道结构信息才能接受读取时的返回值。同时方便的一点就是，
    如果你知道结构是啥样的，返回时就不需要指定读取大小、分段之类的参数，存的时候是什么类型的变量读的时候就用什么变量承接。
    看这个返回值，存的时候的结构应该是((list,list),(list,list),(list,list))，一个大元组套三个小元组，元素则是list。
    """

    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")  # 官方文档写的是"latin1"
    print(x_train)
    # map() 会根据提供的函数对指定序列做映射。
    # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
    # 所有list都被变成tensor了。
    
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    
    from torch.utils.data import TensorDataset

    # 把数据和标签合并到一起。可别写到上面去，函数输入要求是tensor。写到上面最糟的就是这行是不报错的，
    # 照常处理没转成tensor的list，但后面用到这个tensor的代码就该报错了，你想改都不知道是这里错了。
    from torch.utils.data import DataLoader

    train_ds = TensorDataset(x_train, y_train)  # 把训练集数据以dataset的形式存起来，同时合并数据和标签
    valid_ds = TensorDataset(x_valid, y_valid)  # 把验证集数据以dataset的形式存起来，同时合并数据和标签

    # DataLoader里有用sampler的，但是我们打乱了训练集，就没必要用sampler再次打乱了
    def get_data(train_ds, valid_ds, bs):
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True),  # shuffle用于打乱训练数据顺序以防止过拟合。
            DataLoader(valid_ds, batch_size=bs * 2),  # 验证集打乱没意义。验证不需要反传，省下来的内存扩大batch。
        )
    
    return get_data(train_ds, valid_ds, bs)