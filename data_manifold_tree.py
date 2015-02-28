import numpy
import scipy.linalg as LA


class DataManifoldTree(object):
    
    def __init__(self,data):
        """ data is a numpy.ndarray containing the sample points as columns.
            That is, data.shape=(n,D) means there are n sample points in RR^D.
        """
 
        if not ( isinstance(data,numpy.ndarray) and len(data.shape) == 2 ):
            raise TypeError("The source data must be a valid for 2d ndarray")

        self.data=data
        self.samples,self.measurements=self.data.shape

        
    def random_pt(self):
        """ select a random point from our data """
        return self.data[numpy.random.randint(self.samples)]

    def sample_pt(self):
        """ Generate a sample point from our data """

    
    def __str__(self):
        return "DataManifoldTree with {} samples of {} measurements".format(self.samples,self.measurements)
        

    def neighbors(self,point,radius=-1):
        """ return ndarray whose rows are neighbors of point within radius """
    
    def svd(self, point, subset=None):
        """ perform svd on data set versus a point.  
        Can restrict data to a subset given as rows of an ndarray. 
        input point is used as the centroid of the new data.  """ 

        if not subset: subset=self.data

        ## Do I want to ASSUME point is the center, or allow translations?
        import scipy.linalg
        subsamples=subset.shape[0]
        affine=subset - numpy.tile(point,(subsamples,1))
        return scipy.linalg.svd(affine)


if __name__ == "__main__":
    print "Hello"
    data = numpy.array([[1,0,0],[2,0,0],[-1,2,3],[0,0,1],[0,0,-2]])
    A = DataManifoldTree(data)
    print A
    p=A.random_pt()
    print p
    U,s,V=A.svd(p)
    print s


