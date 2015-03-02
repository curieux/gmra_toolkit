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


    
    def __init__(self,data,length=scipy.spatial.distance.cdist):
        """ data is a numpy.ndarray containing the sample points as rows.
            That is, data.shape=(n,D) means there are n sample points in RR^D.
        """
 
        DataManifoldTree.valid_data(data)

        self.data=data
        self.samples,self.instruments=self.data.shape

        self.length=length
        self.distances=self.length(self.data,self.data)
        self.diameter=numpy.amax(self.distances)     
        ### this only works for Euclidean distance!!!
        self.tree=scipy.spatial.cKDTree(self.data)

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
        return "DataManifoldTree with {} samples of {} instruments".format(self.samples,self.instruments)
        

    def view_from_indices(self,indices):
        """ make a mask of the data from a list of "good" indices """
        nbrs=self.data.view(numpy.ma.MaskedArray)
        mask = [ [True]*self.instruments for j in range(self.samples) ]
        for i in indices:
            mask[i]=[False]*self.instruments
        nbrs.mask=numpy.array(mask)
        assert nbrs.mask.shape == self.data.shape
        return nbrs

    def copy_from_indices(self,indices):
        """ make a mask of the data from a list of "good" indices """
        nbrs=self.data[tuple(indices),:]
        return DataManifoldTree(nbrs)

    def neighbors(self,i,radius):
        """ list rows in self.data that are neighbors of sample within radius.

            radius defaults to -1, meaning `infinity' (really `2*diameter')

            We return a masked ndarray view instead of an ndarray because otherwise fancy
            indexing would force copies, not views, and self.data is probably huge.
            """

        if radius==-1: radius=2*self.diameter
        return self.tree.query_ball_point(self.data[i], radius)


    def svd(self, point=None, centroid='point'): #maskedview=None,valsonly=True):
        """ perform svd on data set (versus a point).  
        If mean=='point' (default), input point is used as the centroid of the new data. 
        If mean=='mean', then the mean of the data is used as the centroid.
        If mean=='none', then the data is used on its own, in the ambient coordinates.
        """ 

        import scipy.linalg

        if centroid=='point' and isinstance(point,numpy.ndarray) \
          and point.ndim==1 and point.shape==(self.instruments,):
            centroidv=point

        elif centroid=='mean':
            centroidv=self.data.mean(axis=0) 
        elif centroid=='none':
            centroidv=0
        else:
            raise ValueError("centroid must be 'mean', 'none', or 'point'.  If 'point', provide a point as an ndarray!")

        affine=(1/numpy.sqrt(self.samples -1))*(self.data - centroidv)
        return scipy.linalg.svdvals(affine)

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
    s=A.svd(p)
    print s
#    print A.tree.query(p, 2)
    print A.distances
   
   
    #print A.diameter
    #b=numpy.array([0,0,1])
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

    #import matplotlib.pyplot
        #d=matplotlib.pyplot.scatter(data[:,0],data[:,1])
    #d.show()
    #print data.shape
    print "Now, the slow part"
    C=DataManifoldTree(data)
    #print C.data.shape
    #print C.distances.shape
    print C.diameter
    print C.distances.mean()
#
    j,p=C.random_pt() #0,C[0]
    print p
    for i in [1,5,20,50,100,200,250]:
        print 
#        print 0.1*i
        pn=C.neighbors(j,0.1*i)
        pnp=C.copy_from_indices(pn)
        pd=numpy.array( [ C.distances[j,k] for k in pn ])
#        print "#{}: {} -- {}".format(len(pn), numpy.amin(pd), numpy.amax(pd))
#        print scipy.linalg.svdvals(pnp)
        #bias=numpy.amax(maskeddistances)     
#        print pnp.diameter
        s=pnp.svd(p)
        print 0.1*i, s

    CC=1/float(C.samples-1)*(C.data.T).dot(C.data)
    print numpy.sqrt(scipy.linalg.eig(CC)[0])
    print C.diameter
    print C.svd(centroid='mean')
#,pnp)
#        print type(pnp), pnp.count
        #C.distances[j, (pn) ]), 
        #, numpy.amax(C.distances[
            #print C[idx]
        #affine=(pn - numpy.tile(p,(C.samples,1)))
        #print pn[~pn.mask]
        #print C.svd(p,pn)


    #print pn.shape
    #print pn.mask
#print pn[~pn.mask]
#for index, i, in numpy.ndenumerate(pn[~pn.mask]): 
#        print index, i, k

#    for q in pn:
#        print q
    #    print p
#    print pn.mean()
    #    print 
#print C.i

#    D=C.data.dot(C.data.transpose())
#    for i in range(N):
#        for j in range(N):
#            if D[i,j] < 0: print D[i,j]
#        print numpy.sqrt(D)

