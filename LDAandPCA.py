#this file
#    Implements PCA after that LDA
#    Implements LDA after that PCA
#    Implements PCA only
from scipy import misc
import os, copy, random, math, Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#rnd =random.sample(range(1,11),5)
rnd = [2,4,7,10,9]
path = 'G:\\6363\\orl_faces\\'
folder_list = os.listdir(path)
flag_1 = 0
flag_2 = 0
training_class = []
testing_class = []
def display_me(p,n,shape,title):
    a = int(n/10)
    b = 10
    p = p.transpose()
    plt.figure()
    for i in range(a):
        for j in range(b):
            if j == 0:
                e_print = p[j+i*b,:].reshape(shape)
            else:
                e_print  = np.hstack((e_print, p[j+i*b,:].reshape(shape)))
        if i == 0:
            tot = np.copy(e_print)
        else:
            tot = np.vstack((tot, e_print))     
    #misc.imsave('eigen.png', tot)
    plt.imshow(tot, interpolation='none',cmap='gray')
    plt.title(title)
    plt.show()
    
# !_NN algorithm------------------------------
def K_NN(train, tr_class,test,tst_class):
    cntr = 0
    for i in range(test.shape[1]):# i counts the testing
        dis = []
        for j in range(train.shape[1]):# J counts the trianing 
            dis.append(np.linalg.norm(train[:,j] - test[:,i]))
        inx_tr = dis.index(min(dis))
        if tr_class[inx_tr] == tst_class[i]:
            cntr = cntr + 1
    print "K_NN accuracy rate = {}".format(float(cntr)/float(len(tst_class)))
    return (float(cntr)/float(len(tst_class)))*100.0


#2_fold cross validation:
def two_fold(X,training_class,T,testing_class, shape):
    r = (float(K_NN(X,training_class,T,testing_class))+float(K_NN(T,testing_class, X,training_class)))/2.0    
    print "2 fold cross validation accuracy rate is ", r
    return r
        
            
# PCA algorithm--------------------------------        
def PCA(e_vector,thr,X, tr_class,T,tst_class):      
    cntr = 0
    im_mean = X.mean(axis = 1)# mean face
    # subtracting all faces from mean
    reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
    X_ = X - reshaped_mean
    p = e_vector.transpose()
    p = p[0:thr,:].copy()
    Y = np.dot(p, X_)
    T_ = T - (T.mean(axis = 1)).reshape((-1,1))
    TY = np.dot(p, T_)
    for i in range(TY.shape[1]):# i counts the testing
        dis = []
        for j in range(Y.shape[1]):# J counts the trianing 
            dis.append(np.linalg.norm(Y[:,j] - TY[:,i]))
        inx_tr = dis.index(min(dis))
        if tr_class[inx_tr] == tst_class[i]:
            cntr = cntr + 1
    print "PCA accuracy rate = {}".format((float(cntr)/float(len(tst_class)))*100.0)   
    return (float(cntr)/float(len(tst_class)))* 100.0
    
    
    
    
#--------------------reading images from HD-----------------
def open_images(path,folder_list,siz = None):
    #print "Reading images....."
    if siz is not None:
        print "   Images will be resized by ", siz
    flag_1 = 0
    flag_2 = 0
    training_class = []
    testing_class = []
    for folder in folder_list:
        file_path =path + folder
        file_list = os.listdir(file_path)
        for i in range(1,11):
            im = misc.imread(file_path + '\\' + str(i) + '.pgm')
            if siz is not None:
                im = misc.imresize(im, siz)
            shape = im.shape
            im = im.reshape((im.shape[0]*im.shape[1], 1)) 
            if i in rnd:#-------------Training matrix-----------------
                if flag_1 == 0:
                    X = np.copy(im)
                    flag_1 = 1
                    training_class.append(folder)
                else:
                    X  = np.hstack((X, np.copy(im)))
                    training_class.append(folder)
            else:#--------testing matrix--------------
                if flag_2 == 0:
                    T = np.copy(im)
                    flag_2 = 1
                    testing_class.append(folder)
                else:
                    T  = np.hstack((T, np.copy(im)))  
                    testing_class.append(folder)
    print '   We read {0} images as training and {1} images as testing and saved them in matrixes in  ({1} x {2})'.format(X.shape[1],T.shape[1], X.shape[0],X.shape[1])
    #display_me(X, X.shape[1],shape, "Original faces")
    #X1 = X.astype(np.float, copy = True)
    #T1 = T.astype(np.float, copy = True)
    Xclasses = []
    Tclasses =[]
    for c in folder_list:
        s = np.empty([X.shape[0],0])
        for clm in range(X.shape[1]):
            if (c == training_class[clm]):
                s = np.hstack((s, X[:,clm].reshape((X.shape[0],1))))
        Xclasses.append(s)
        st = np.empty([T.shape[0],0])
        for clm in range(T.shape[1]):
            if (c == testing_class[clm]):
                st = np.hstack((st, T[:,clm].reshape((T.shape[0],1))))
        Tclasses.append(st) 

    return X,training_class,T,testing_class, shape, Xclasses, Tclasses



def get_classes(X,training_class,folder_list):
    Xclasses = []
    for c in folder_list:
        s = np.empty([X.shape[0],0])
        for clm in range(X.shape[1]):
            if (c == training_class[clm]):
                s = np.hstack((s, X[:,clm].reshape((X.shape[0],1))))
        Xclasses.append(s)
    return Xclasses


# eigenvalues and eigenvectors
def eigen(X, shape= None):
    #print "Eigenvalues and eigenvectors finder: Running...."
    im_mean = X.mean(axis = 1)# mean face
    if shape is not None:
        dis_im_mean = im_mean.reshape(shape) # reshape to display it
        misc.imsave('mean.png', dis_im_mean)
    # subtracting all faces from mean
    reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
    #print "   Subtracting mean face from matrix X ..."
    X_ = X - reshaped_mean
    # transpose of matrix------------------------
    X_T = X_.transpose()
    #print "   Transposing matrix X, the shape will be {0}x{1} ".format(X_T.shape[0],X_T.shape[1]) 
    
    #---muliplying A^T and A-----------------
    mult = np.dot(X_T,X_)/float(X.shape[1] - 1)
    
    #print "   Computing coveriance matrix (X^T * X)/({0} - 1), the shape would be {0}x{1} ".format(X.shape[1],mult.shape[0],mult.shape[1]) 
    
    #eigenvalues and eigenvectors---------
    e_values, e_vectors = np.linalg.eigh(mult)
    #idx = e_values.argsort()[::-1]
    idx = np.argsort(-e_values)
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]
    #print "   Computing eigenvalues and eigenvectors and sort them descendingly, eigenvectors is {0}x{1} ".format(e_vectors.shape[0],e_vectors.shape[1]) 
    #---------------------------------------------
    e_vectors_ = np.zeros((X.shape[0],e_vectors.shape[0]))# an eigen vectors matrix N by M ---eigenvectors of XX^T
    for i in range(e_vectors.shape[1]):
        e_vectors_[:,i] = np.dot(X_, e_vectors[:,i])
        e_vectors_[:,i] = e_vectors_[:,i]/np.linalg.norm(e_vectors_[:,i])
    
    #print "   Computing X * Eigenvectors, eigenvectors is {0}x{1} and normalizing them ".format(e_vectors_.shape[0],e_vectors_.shape[1])
    #----------------------------------------------------------
    eig_list = e_values.tolist()
    plt.figure()
    pl = plt.plot(range(1,len(eig_list)+1),eig_list)
    plt.title("all eigenvectors")
    plt.ylabel('Values')
    plt.xlabel('the number of eigenvectors')
    plt.grid(True)
    plt.show()
    return e_values, e_vectors_
#Starting---------------------------------------------    


def get_Sw(Xclasses, shape, siz = None):
    sz = shape[0]*shape[1]
    if siz is not None:
        sz = siz
    Sw = np.zeros((sz,sz))
    mean_tot = np.zeros((sz,1))
    mean_list = []
    for c in Xclasses:
        m = c.mean(axis = 1).reshape((-1,1))
        mean_list.append(m)
        mean_tot = mean_tot + m
        for cl in range(c.shape[1]):
            Sw = Sw + np.dot((c[:,cl].reshape((-1,1)) - m), ((c[:,cl].reshape((-1,1)) - m)).T)
    mean_tot =mean_tot/float(len(Xclasses))
    Sb = np.zeros((sz,sz))
    for m in mean_list:
        Sb = Sb + (np.dot((m - mean_tot),(m - mean_tot).T)*float(5))
            
    return Sw, Sb

# Here we compute the w transforer matrix for LDA
def LDA_w(Xclasses, shape, N = None):
    Sw,Sb = get_Sw(Xclasses,shape, Xclasses[0].shape[0])
    #print "   Sw's shape:", Sw.shape," , Sb's shape:  ", Sb.shape
    Sw_inv = np.linalg.inv(Sw)
    e_values,e_vectors = np.linalg.eigh(np.dot(Sw_inv,Sb))
    #print "   number of eigenvalues:  ", len(e_values)
    idx = np.argsort(-e_values)
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]
    d = len(Xclasses)    
    if N is not None:
        d = N
    w = e_vectors[:,0:(d - 1)].copy()
    w = w.transpose()
    return w
    

# PCA algorithm--------------------------------        
def LDA_on_PCA(e_vector,thr,X, tr_class,T,tst_class,shape):      
    cntr = 0
    im_mean = X.mean(axis = 1)# mean face
    # subtracting all faces from mean
    reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
    X_ = X - reshaped_mean
    p = e_vector.transpose()
    p = p[0:thr,:].copy()
    Y = np.dot(p, X_)
    T_ = T - (T.mean(axis = 1)).reshape((-1,1))
    TY = np.dot(p, T_)
    Yclasses = get_classes(Y,tr_class,folder_list)
    #TYclasses = get_classes(TY,tst_class,folder_list)    
    w = LDA_w(Yclasses, shape)
    #print "w's shape: ", w.shape
    Y_LDA = np.dot(w,Y)
    T_LDA = np.dot(w,TY)
    for i in range(T_LDA.shape[1]):# i counts the testing
        dis = []
        for j in range(Y_LDA.shape[1]):# J counts the trianing 
            dis.append(np.linalg.norm(Y_LDA[:,j] - T_LDA[:,i]))
        inx_tr = dis.index(min(dis))
        if tr_class[inx_tr] == tst_class[i]:
            cntr = cntr + 1
    #print "PCA accuracy rate = {}".format((float(cntr)/float(len(tst_class)))*100.0)   
    return (float(cntr)/float(len(tst_class)))* 100.0
        
X,training_class,T,testing_class, shape, Xclasses, Tclasses = open_images(path,folder_list,(40,30))
e_values, e_vectors = eigen(X)
thres = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,110,120]
LDAonPCA =[]
for i in thres:
    LDAonPCA.append(LDA_on_PCA(e_vectors,i,X, training_class,T,testing_class,shape))
    
    
#---------------------------------------------------------------------------   
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
if True:        
    #Get matrix W from LDA
    w = LDA_w(Xclasses, shape)    
    #The training  and testing set go to the new space by LDA
    X_LDA = np.dot(w,X)
    T_LDA = np.dot(w,T)
    #Run PCA over LDA---------------------------------------
    e_values, e_vectors = eigen(X_LDA)
    PCA_on_LDA=[]
    for i in thres:
        PCA_on_LDA.append(PCA(e_vectors,i,X_LDA, training_class,T_LDA,testing_class))
    print "\n\n\n\t\t We are doing PCA only\n\n\n"      
    #Runing the PCA only ---------------------------------------------------------
    #Runing the PCA only ---------------------------------------------------------
    e_values, e_vectors_ = eigen(X)
    PCA_only = []
    for i in thres: 
        PCA_only.append(PCA(e_vectors_, i, X,training_class, T, testing_class))
        
    
    plt.figure()
    pp = PdfPages('LDAandPCA.pdf')    
    pl1 = plt.plot(thres, PCA_only,color = 'r', label = "PCA only")
    pl2 = plt.plot(thres, PCA_on_LDA,color = 'b', label = "PCA over LDA")
    pl3 = plt.plot(thres, LDAonPCA,color = 'g', label = "LDA over PCA")
    
    plt.legend( loc='lower right', numpoints = 1 )
    plt.title(' PCA and LDA, image size ' + str(shape))
    plt.ylabel('Accuracy rate (%)')
    plt.xlabel('the number of most segnificant eigenvectors')
    plt.grid(True)
    plt.show()
    plt.savefig(pp, format='pdf')
    pp.close()
