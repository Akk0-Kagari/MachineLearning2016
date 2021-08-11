import math
import numpy as np

class LinearRegression(object):
    def __init__(self):
        self.__wt = None
        self.__b = None
        self.__x = None
        self.__y = None

    @property
    def wt(self):
        return self.__wt

    @property
    def b(self):
        return self.__b


    def train_by_pseudo_inverse(self,x, y, alpha=0,validate_data=None):
        self._check_data(x,y)

        self.__x=x
        self.__y=y

        num_data = x.shape[0]
        dim_wt = x.shape[1]

        argu_X = np.hstack((np.ones(num_data).reshape(num_data,1),x))

        A_plus = np.dot(np.linalg.inv(np.dot(argu_X.T,argu_X)+alpha*np.eye(dim_wt+1)),argu_X.T)

        wt_b = np.dot(A_plus,y)
        self.__b=wt_b[0]
        self.__wt = wt_b[1:]

        if validate_data:
            print("Pseudo-Inverse: err = {:.6f} validate = {:.6f}".format(
                self.err_insample(),
                self.err(validate_data[0],validate_data[1])
            ))
        else:
            print("Pseudo-Inverse: err = {:.6f}".format(self.err_insample()))

    def train_by_gradient_descent(self,x, y, init_wt=np.array([]),init_b=0,
                                  rate=0.01, alpha=0,epoch=1000,batch=None, validate_data = None):
        self._check_data(x,y)

        if init_wt.size == 0:
            init_wt = np.zeros(x.shape[1])


        self.__x = x
        self.__y = y

        self.__wt = init_wt
        self.__b = init_b

        # rows of data
        num_data = x.shape[0]

        if not batch:
            batch = num_data

        tot_batch = int(math.ceil(float(num_data)/float(batch)))

        if validate_data:
            for i in range(epoch):
                for j in range(tot_batch):
                    batch_x = self.__x[j*batch : min(num_data, (j+1)*batch), :]
                    batch_y = self.__y[j*batch : min(num_data,(j+1)*batch)]
                    self.__b, self.__wt = self._gd_update(batch_x, batch_y, self.__wt,self.__b,rate,alpha)
                print("Epoch {:5d}: err = {:.6f} validate = {:.6f}".format(
                    i+1,
                    self.err_insample(),
                    self.err(validate_data[0],validate_data[1])
                ))

    def _gd_update(self,x, y ,wt,b,rate,alpha):
        num_data = x.shape[0]

        # y_pred
        y_pred = np.sum(wt * x,axis=1) + b
        # y_pred - y
        y_diff = y_pred - y

        # b update
        b_gradient = np.sum(y_diff)/num_data
        new_b = b - rate * b_gradient
        #wt update
        new_wt = []
        for i,w in enumerate(wt):
            w_gradient = np.sum(x[:,i]*y_diff)/num_data + alpha * w
            new_wt.append(w - rate*w_gradient)

        new_wt = np.array(new_wt,dtype='float64')
        return (new_b,new_wt)

    def err_insample(self):
        if self.__x.size == 0 or self.__y.size == 0:
            raise RuntimeError('in-sample data not found')

        return self.err(self.__x,self.__y)

    def err(self,x,y):
        self._check_data(x,y)

        if self.__wt.size == 0 or self.__b == 0:
            raise RuntimeError("model haven't been trained!")
        y_pred = np.sum(self.__wt*x,axis=1) + self.__b

        err = (np.sum((y-y_pred)**2)/y.shape[0]) ** 0.5
        return round(err,8)

    def _check_data(self,x,y):
        if x.shape[0] != y.shape[0]:
            raise ValueError('shape of x and y do not match')


    def predict(self,x):
        if type(x) == list:
            x= np.array(x, dtype="float64")

        if x.shape[1] != self.__wt.shape[0]:
            raise ValueError("shape of input x does not match shape of weight")

        return np.sum(self.__wt * x,axis=1) + self.__b

