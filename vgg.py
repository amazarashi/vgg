import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils

class VGG(chainer.Chain):

    def __init__(self,category_num=10):
        initializer = chainer.initializers.HeNormal()
        super(VGG,self).__init__(
            conv1_1 = L.Convolution2D(3,64,3,1,1,initialW=initializer),
            conv1_2 = L.Convolution2D(64,128,3,1,1,initialW=initializer),
            conv2_1 = L.Convolution2D(128,128,3,1,1,initialW=initializer),
            conv2_2 = L.Convolution2D(128,256,3,1,1,initialW=initializer),
            conv3_1 = L.Convolution2D(256,256,3,1,1,initialW=initializer),
            conv3_2 = L.Convolution2D(256,256,3,1,1,initialW=initializer),
            conv3_3 = L.Convolution2D(256,256,3,1,1,initialW=initializer),
            conv3_4 = L.Convolution2D(256,512,3,1,1,initialW=initializer),
            conv4_1 = L.Convolution2D(512,512,3,1,1,initialW=initializer),
            conv4_2 = L.Convolution2D(512,512,3,1,1,initialW=initializer),
            conv4_3 = L.Convolution2D(512,512,3,1,1,initialW=initializer),
            conv4_4 = L.Convolution2D(512,512,3,1,1,initialW=initializer),
            fc1 = L.Linear(25088,4096),
            fc2 = L.Linear(4096,4096),
            fc3 = L.Linear(4096,1000),
            fc4 = L.Linear(1000,10),
        )

    def __call__(self,x,train=True):
        #x = chainer.Variable(x)
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv3_1(x))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(x))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv4_1(x))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(x))
        h = F.relu(self.conv4_4(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.fc1)
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc2)
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc3)
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc4)
        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy

class VGG_A(chainer.Chain):

    def __init__(self,category_num=10):
        initializer = chainer.initializers.HeNormal()
        super(VGG_A,self).__init__(
            conv1_1 = L.Convolution2D(3,64,3,1,1,initialW=initializer),
            conv2_1 = L.Convolution2D(64,128,3,1,1,initialW=initializer),
            conv3_1 = L.Convolution2D(128,256,3,1,1,initialW=initializer),
            conv3_2 = L.Convolution2D(256,256,3,1,1,initialW=initializer),
            conv4_1 = L.Convolution2D(256,512,3,1,1,initialW=initializer),
            conv4_2 = L.Convolution2D(512,512,3,1,1,initialW=initializer),
            fc1 = L.Linear(25088,4096),
            fc2 = L.Linear(4096,4096),
            fc3 = L.Linear(4096,1000),
            fc4 = L.Linear(1000,10),
        )

    def __call__(self,x,train=True):
        #x = chainer.Variable(x)
        h = F.relu(self.conv1_1(x))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv2_1(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.max_pooling_2d(h,2,stride=2,pad=1)

        h = F.relu(self.fc1(h))
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc2(h))
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc3(h))
        h = F.dropout(h,ratio=0.5,train=train)
        h = F.relu(self.fc4(h))
        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy

if __name__ == "__main__":
    imgpath = "/Users/suguru/Desktop/test.jpg"
    img = io.imread(imgpath)
    img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    img = img[np.newaxis]
    ex = model(img)
    print(ex)
