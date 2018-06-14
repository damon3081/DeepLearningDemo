import copy, numpy as np

# compute sigmoid nonlinearity 我们的激活函数和求导函数。
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def dersigmoid(x):
    y = x*(1-x)
    return y

step = 0.1       #学习步长设置为0.1
ipnumber = 2     #每次我们喂给RNN的输入数据是两个比特
hdnumber = 32    #这是隐层的比特数。也可以说是隐层神经元个数。
opnumber = 2     #输出层我们预测2位求和值
neu_i2h = 2*np.random.random((ipnumber,hdnumber)) - 1   #这是输入层和隐层间的权重矩阵
neu_h2o = 2*np.random.random((hdnumber,opnumber)) - 1   #这是隐层和输出层间的权重矩阵
neu_h2h = 2*np.random.random((hdnumber,hdnumber)) - 1   #这是连接上一个时间戳隐层和当前时间戳隐层的矩阵，同时也是连接当前时间戳隐层和下一个时间戳隐层的矩阵。所以矩阵是隐层单元*隐层单元（32 x 32）

#这些变量保存对于权重矩阵的更新值，我们的目的不就是训练好的权重矩阵吗？我们在每次迭代积累权重更新值，然后一起更新
neu_i2hN = np.zeros_like(neu_i2h)
neu_h2oN = np.zeros_like(neu_h2o)
neu_h2hN = np.zeros_like(neu_h2h)

#i2b和bin都是一个从整数到比特串的表示查找表
i2b = {}
bdigit = 8      #比特串的最大长度
MAXnumber = pow(2,bdigit)
bin = np.unpackbits(np.array([range(MAXnumber)],dtype=np.uint8).T,axis=1)
for i in range(MAXnumber):
    i2b[i] = bin[i]

for j in range(10000):      #迭代训练10000个训练样本

    # 我们将要生成一个随机加和问题。我随机生成的整数不会超过我们所能表达的整数的一半，否则两个整数相加就有可能超过我们可以用比特串表达的整数
    # generate a simple addition problem (a + b = c)
    a_decimal = np.random.randint(MAXnumber / 2)
    b_decimal = np.random.randint(MAXnumber / 2)
    c_decimal = a_decimal + b_decimal    # true answer计算应该得出结果
    a = i2b[a_decimal]    # 查找整数a对应的比特串
    b = i2b[b_decimal]    # 查找整数b对应的比特串
    c = i2b[c_decimal]    # 查找整数c对应的比特串
    binary = np.zeros_like(c)  #得到一个空的比特串来存储络的我们RNN神经网预测值
    aError = 0             #初始化错误估计，作为收敛的依据

    # 这两个列表是在每个时间戳跟踪输出层求导和隐层值的列表
    oplayer_der = list()
    hdlayer_val = list()
    hdlayer_val.append(np.zeros(hdnumber))   #开始时没有上一个时间戳隐层，所有我们置为0

    for locate in range(bdigit):   #这个迭代可能的比特串表达（8位比特串）
        # generate input and output
        # X 是一个2个元素的列表，第一个元素是比特串a中的，第二个元素是比特串b中的。
        # 我们用locate定位比特位，是自右向左的
        X = np.array([[a[bdigit - locate - 1],b[bdigit - locate - 1]]])
        Y = np.array([[c[bdigit - locate - 1]]]).T  #正确结果01串

        # 这行是代码神奇之处!!! 请看懂这一行!!! 为了构造隐层，我们做两件事，第一步是从输入层传播到隐层(np.dot(X,synapse_0))。
        ## 第二步，我们把上一个时间戳的隐层值传播到当前隐层 (np.dot(prev_layer_1, synapse_h)。最后我们把两个向量值相加!
        ## 最后交给sigmoid函数
        # hidden layer (input ~+ prev_hidden)
        hdlayer = sigmoid(np.dot(X,neu_i2h) + np.dot(hdlayer_val[-1],neu_h2h))
        oplayer = sigmoid(np.dot(hdlayer,neu_h2o))   #把隐层传播到输出层，做预测
        oplayer_error = Y - oplayer                  #计算预测的错误偏差
        oplayer_der.append((oplayer_error)*dersigmoid(oplayer))  #计算并存储错误导数，在每个时间戳进行
        aError += np.abs(oplayer_error[0])                       #计算错误的绝对值的和，积累起来

        binary[bdigit - locate - 1] = np.round(oplayer[0][0])   #估计输出值。并且保存在binary中

        hdlayer_val.append(copy.deepcopy(hdlayer))              #保存当前隐层值，作为下个时间戳的上个隐层值

    Fhdlayer_dels = np.zeros(hdnumber)

    # 所以，我们对于所有的时间戳做了前向传播，我们计算了输出层的求导并且把它们存在列表中。
    ## 现在我们需要反向传播，从最后一个时间戳开始反向传播到第一个时间戳
    for locate in range(bdigit):
        X = np.array([[a[locate],b[locate]]])     #像我们之前一样获得输入数据
        hdlayer = hdlayer_val[-locate-1]          #选择当前隐层
        hdlayer_pre = hdlayer_val[-locate-2]      #选择上个时间戳隐层

        oplayer_dels = oplayer_der[-locate-1]     #选择当前输出错误
        # error at hidden layer
        # 这行在给定下一个时间戳隐层错误和当前输出错误的情况下，计算当前隐层错误
        hdlayer_dels = (Fhdlayer_dels.dot(neu_h2h.T) + oplayer_dels.dot(neu_h2o.T)) * dersigmoid(hdlayer)

        # let's update all our weights so we can try again
        neu_h2oN += np.atleast_2d(hdlayer).T.dot(oplayer_dels)
        neu_h2hN += np.atleast_2d(hdlayer_pre).T.dot(hdlayer_dels)
        neu_i2hN += X.T.dot(hdlayer_dels)
        # 以上3行：现在我们在当前时间戳通过反向传播得到了求导，我们可以构造权重更新了（但暂时不更新权重）。
        ## 我们等到完全反向传播后，才真正去更新权重。为什么？因为反向传播也是需要权重的。乱改权重是不合理的
        Fhdlayer_dels = hdlayer_dels

    # 现在我们反向传播完毕，可以真的更新所有权重了
    neu_i2h += neu_i2hN * step
    neu_h2o += neu_h2oN * step
    neu_h2h += neu_h2hN * step


    neu_i2hN *= 0
    neu_h2oN *= 0
    neu_h2hN *= 0

    print("Error:" + str(aError))
    print("Predicted:" + str(binary))
    print("True:" + str(c))
    value = 0
    for index,x in enumerate(reversed(binary)):
        value += x*pow(2,index)
    print(str(a_decimal) + " + " + str(b_decimal) + " = " + str(value))
    print("--------------------------------------")