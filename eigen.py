#This file 
#     Plot the faces, most significant eigenfaces
#     Runs the KNN, 2 fold cross validation
#     Runs PCA

# Please change vairiable 'path' to the location of data set

from scipy import misc
import os, copy, random, math, Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ppp = PdfPages('baiscinfo.pdf')    

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
    plt.savefig(ppp, format='pdf')

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
    print "PCA accuracy rate = {}".format(float(cntr)/float(len(tst_class)))   
    return (float(cntr)/float(len(tst_class)))* 100.0
    
    
    
    
#--------------------reading images from HD-----------------
def open_images(siz = None):
    print "Reading images....."
    if siz is not None:
        print "   Images will be resize by ", siz
    path = 'G:\\6363\\orl_faces\\'
    folder_list = os.listdir(path)
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
    display_me(X, X.shape[1],shape, "Original faces")
    #X1 = X.astype(np.float, copy = True)
    #T1 = T.astype(np.float, copy = True)
    return X,training_class,T,testing_class, shape

# eigenvalues and eigenvectors
def eigen(X, shape):
    print "Eigenvalues and eigenvectors finder: Running...."
    im_mean = X.mean(axis = 1)# mean face
    dis_im_mean = im_mean.reshape(shape) # reshape to display it
    misc.imsave('mean.png', dis_im_mean)
    # subtracting all faces from mean
    reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
    print "   Subtracting mean face from matrix X ..."
    X_ = X - reshaped_mean
    # transpose of matrix------------------------
    X_T = X_.transpose()
    print "   Transposing matrix X, the shape will be {0}x{1} ".format(X_T.shape[0],X_T.shape[1]) 
    
    #---muliplying A^T and A-----------------
    mult = np.dot(X_T,X_)/float(X.shape[1] - 1)
    
    print "   Computing coveriance matrix (X^T * X)/({0} - 1), the shape would be {0}x{1} ".format(X.shape[1],mult.shape[0],mult.shape[1]) 
    
    #eigenvalues and eigenvectors---------
    e_values, e_vectors = np.linalg.eigh(mult)
    #idx = e_values.argsort()[::-1]
    idx = np.argsort(-e_values)
    e_values = e_values[idx]
    e_vectors = e_vectors[:,idx]
    print "   Computing eigenvalues and eigenvectors and sort them descendingly, eigenvectors is {0}x{1} ".format(e_vectors.shape[0],e_vectors.shape[1]) 
    #---------------------------------------------
    e_vectors_ = np.zeros((X.shape[0],e_vectors.shape[0]))# an eigen vectors matrix N by M ---eigenvectors of XX^T
    for i in range(e_vectors.shape[1]):
        e_vectors_[:,i] = np.dot(X_, e_vectors[:,i])
        e_vectors_[:,i] = e_vectors_[:,i]/np.linalg.norm(e_vectors_[:,i])
    
    print "   Computing X * Eigenvectors, eigenvectors is {0}x{1} and normalizing them ".format(e_vectors_.shape[0],e_vectors_.shape[1])
    #----------------------------------------------------------
    eig_list = e_values.tolist()
    plt.figure()
    pl = plt.plot(range(1,len(eig_list)+1),eig_list)
    plt.title("all eigenvectors, size " + str(shape))
    plt.ylabel('Values')
    plt.xlabel('the number of eigenvectors')
    plt.grid(True)
    plt.show()
    plt.savefig(ppp, format='pdf')

    return e_values, e_vectors_
#Starting---------------------------------------------    

print '2 fold validation: Running with original size...........'
#K_NN(train, tr_class,test,tst_class)
X,training_class,T,testing_class, shape = open_images()
a = two_fold(X,training_class,T,testing_class, shape)       
print '2 fold validation: Running images resized to {}'.format((56,46))
#K_NN(train, tr_class,test,tst_class)
X,training_class,T,testing_class, shape = open_images((56,46))
b = two_fold(X,training_class,T,testing_class, shape) 
# Now I wanna compute the mean face, matrix big is 10304 X 200
print "\n\n \t\ttwo-fold cross validation in original size = ", a
print "\t\ttwo-fold cross validation as images resized by (56,46) = ", b, "\n\n"

#---------------------------------------------------------
#PCA(e_vector,thr,X_, tr_class,T,tst_class)
X,training_class,T,testing_class, shape = open_images()

e_values, e_vectors_ = eigen(X,shape)
print 'PCA Running....'
plt.figure()
thres = [1,5,10,20,30,40,50,70,100,150,200]
ret = []
for i in thres: 
    ret.append(PCA(e_vectors_, i-1, X,training_class, T, testing_class))
pl = plt.plot(thres, ret)
plt.title("PCA accuracy rate for different number of most segnificant eigenvectors")
plt.ylabel('Accuracy rate (%)')
plt.xlabel('the number of most segnificant eigenvectors')
plt.grid(True)
plt.show()
plt.savefig(ppp, format='pdf')

#----------------------------------------------------------
im_mean = X.mean(axis = 1)# mean face
    # subtracting all faces from mean
reshaped_mean = im_mean.reshape((im_mean.shape[0],1))
X_ = X - reshaped_mean
ther = 50
display_me(X,ther,shape, "{0} Original Images".format(ther))
display_me(e_vectors_,ther,shape, "{0} most significant eigenvectors".format(ther))
print "Displaying {0} most significant eigenvectors...".format(ther)
print "Reconstracting based on all basis vectors..."
p = e_vectors_.transpose()
Y = np.dot(p,X_)
X_recon = np.dot(p.T, Y) + reshaped_mean
display_me(X_recon,50, shape,"Reconstruction of {0} Images, using ALL basis vectors.".format(ther))

print "Reconstracting based on {0} basis vectors.".format(ther)
pp = p[0:ther,:].copy()
Y = np.dot(pp,X_)
X_recon = np.dot(pp.T, Y) + reshaped_mean
display_me(X_recon,50, shape,"Reconstruction of {0} Images, using {1} basis vectors.".format(ther, ther))
plt.savefig(ppp, format='pdf')
ppp.close()