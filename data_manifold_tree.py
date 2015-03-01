import numpy
import scipy.linalg 
import scipy.spatial

class DataManifoldTree(object):

    @staticmethod
    def valid_data(data):
        """ simple check for well-posed 2d numerical data. """
        if not ( isinstance(data,numpy.ndarray) and data.ndim == 2 ):
            raise TypeError("The source data must be a valid numpy.ndarray with ndim==2")
        
        ### eventually, also test for dtype, and NaN
        return True


    
    def __init__(self,data,length=scipy.spatial.distance.pdist):
        """ data is a numpy.ndarray containing the sample points as rows.
            That is, data.shape=(n,D) means there are n sample points in RR^D.
        """
 
        DataManifoldTree.valid_data(data)

        self.data=data
        self.samples,self.measurements=self.data.shape

        self.length=length
        self.vdistances=self.length(self.data)
        self.diameter=numpy.amax(self.vdistances)     
        self.distances=scipy.spatial.distance.squareform(self.vdistances)



    def random_pt(self):
        """ select a random point from our data """
        i=numpy.random.randint(self.samples)
        return (i,self.data[i])

    def sample_pt(self):
        """ _Generate_ a sample point from our data, using dictionary compression. """
        
    def find(self,sample,tolerance=0):
        """ Get the indices of matching samples.  Tolerance is allowed using self.length"""
        return tuple( [i for i in xrange(self.samples)\
            if self.length(self.data[i],sample) <= tolerance])


    def __getitem__(self,key):
        return self.data.__getitem__(key)

    def __str__(self):
        return "DataManifoldTree with {} samples of {} measurements".format(self.samples,self.measurements)
        

    def neighbors(self,i,radius=None):
        """ list rows in self.data that are neighbors of sample within radius.

            radius defaults to None, meaning `infinity'

            We return a list instead of an ndarray because otherwise fancy
            indexing would force copies, not views, and self.data is probably huge.
            Feed this output to numpy.array( ) in case you want to operate on
            it, at the cost of memory. 
            """
        if not radius: radius=2*self.diameter  ## catch EVERYTHING 
        dist_from_i=self.distances[i]
        return [ self[j] for j in range(self.measurements) if self.distances[i][j] <= radius ]
#         return [ self[j] for i in self.find(sample,tolerance=radius) ]
            

    def svd(self, point, subset=None):
        """ perform svd on data set versus a point.  
        Can restrict data to a subset given as rows of an ndarray. 
        input point is used as the centroid of the new data.  """ 

        if not subset: subset=self.data

        ## Do I want to ASSUME the point is the center, or allow translations?
        import scipy.linalg
        subsamples=subset.shape[0]
        affine=subset - numpy.tile(point,(subsamples,1))
        return scipy.linalg.svd(affine)


if __name__ == "__main__":
    print "Hello"
    data = numpy.array([[1,0,0],[2,0,0],[-1,2,3],[0,0,1],[0,0,-2]])
    A = DataManifoldTree(data)
    print A
    #A.calculate()
    print A.samples
    print A.data
    i,p=A.random_pt()
    print A.data is p.base
    U,s,V=A.svd(p)
    print s
    print A.distances
    print A.diameter
    b=numpy.array([0,0,1])
    #print A.find(b,2)
    #An=A.neighbors(b,2)
    #print type(An[0].base)
    #print type(An[1].base)


    print "Now, let's generate some real data!"
    data=[]
    N=10000
    for i in xrange(N):
        theta=i*2*numpy.pi/N
        data.append([ 10*numpy.cos(theta)+numpy.random.rand(), 
            10*numpy.sin(theta)+numpy.random.rand()])
    data=numpy.array(data)

    import matplotlib.pyplot
    d=matplotlib.pyplot.scatter(data[:,0],data[:,1])
    d.show()
    print data.shape
    print "Now, the slow part"
    C=DataManifoldTree(data)
    print C.data.shape
    print C.distances.shape
    print C.diameter
    print C.distances.mean()

    for i in xrange(5):
        j,p=C.random_pt()
        pn=C.neighbors(j,5.0)
        print
        print p
        print pn
        print 
#print C.i

#    D=C.data.dot(C.data.transpose())
#    for i in range(N):
#        for j in range(N):
#            if D[i,j] < 0: print D[i,j]
#        print numpy.sqrt(D)

