import numpy
import scipy.linalg as LA

class DataManifoldTree(object):
    
    def __init__(self,data):
        """ data is a numpy.array containing the sample points as columns.
            That is, data.shape=(D,n) means there are n sample points in RR^D.
        """

        if not isinstance(data_array,numpy.array) or not len(data.shape) == 2:
            raise TypeError("The source data must be a 2d numpy.array")


        self.data=data
        (self.n,self.D)=self.data.shape



#    def get_point(self):
#        """


        

if __name__ == "__main__":
    print "Hello"
