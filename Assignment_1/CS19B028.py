# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:16:22 2022

@author: jayak
"""
#Name : Maddula Jaya Kamal
#Roll Number : CS19B028
#Task : Assignment-1
#Description : Representing an image as matrix, perforrming EVD & SVD and reconstructing the image.

#importing the required function from their respective libraries
#importing the required function from their respective libraries
from PIL import Image as img
from matplotlib import pyplot
import numpy as np
import os


def EVD(imgMatrix): #Eigen Value Decomposition Function
    eigVals, eigVecs = np.linalg.eig(imgMatrix) #Getting the Eigen Values and Vectors
    eigValMatrix = np.zeros((256, 256), dtype=complex)  #Getting a null matrix
    sortKey = np.argsort(-1*(np.absolute(eigVals))) #Sorting in descending order of magnitudes 
    eigVals = eigVals[sortKey]  #Sorting Eigen Values 
    eigVecs = eigVecs[:, sortKey] #Sorting Eigen Vectors
    eigVecsInv = np.linalg.inv(eigVecs) #Getting the inverse of eigen vector matrix
    frobNormLst_evd = []
    counter = 1
    if not os.path.exists('evd_images'): #checking if a folder exists with same name
        os.mkdir('evd_images')  #making a folder to save images for all values of k
    if not os.path.exists('ErrorImages_EVD'): #checking if a folder exists with same name
        os.mkdir('ErrorImages_EVD')  #making a folder to save images for all values of k    
    for i in range(256):
        fileName = 'evd_images/'+str(i+1)+'.jpg' #Setting a filename for new image
        eigValMatrix[i][i] = eigVals[i] 
        reconstImgMat_evd = (eigVecs@eigValMatrix)@eigVecsInv #getting provisional product
        tempFrobNorm = np.linalg.norm(imgMatrix-reconstImgMat_evd) #computing the Frob Norm
        frobNormLst_evd.append(tempFrobNorm) #Appending the Frob Norm to a list
        x = np.rint(np.absolute(reconstImgMat_evd))
        tempImg = img.fromarray(x.astype(np.uint8)) #converting the matrix into image
        if i+1 == counter: #plotting reconstructed img vs error image
            pyplot.subplot(1, 2, 1)
            pyplot.imshow(tempImg, cmap="gray")
            plotName = "K = " + str(counter) + "\nReconstructed Image"
            pyplot.title(plotName)
            errorImg = np.rint(np.absolute(imgMatrix-reconstImgMat_evd))
            tempErrImg = img.fromarray(errorImg.astype(np.uint8))
            pyplot.subplot(1, 2, 2)
            pyplot.imshow(tempErrImg, cmap="gray")
            plotName = "K = " + str(counter) + "\nError Image"
            pyplot.title(plotName)
            errfileName = "ErrorImages_EVD/"+str(i+1)+".png"
            if os.path.exists(errfileName):
                os.remove(errfileName)
            pyplot.savefig(errfileName)
            pyplot.close()
            counter = counter*2
            
        if os.path.exists(fileName):
            os.remove(fileName)
        tempImg.save(fileName) #Saving the image for current value of K.
    
    pyplot.show()
    pyplot.close()
    pyplot.plot(frobNormLst_evd) #plotting the required graph
    pyplot.xlabel('K-Value')
    pyplot.ylabel('Frobenius Norm')
    pyplot.title('Frobenius Norm vs K, EVD')
    if(os.path.exists('evd_graph.png')):
        os.remove('evd_graph.png')
    pyplot.savefig('evd_graph.png') #Saving the graph
    pyplot.close() #closing the plotted graph
    
def SVD(imgMatrix):
    A, V = np.linalg.eigh((imgMatrix.T)@imgMatrix) #Decomposing A.(transpose(A))    
    sigma = np.sqrt(np.abs(A))  #computing the square root of eigen values to get sigma
    sigma = sigma[::-1] #sorting sigma in descending order
    V = V[:, ::-1]
    U = imgMatrix@V@np.linalg.inv(np.diag(sigma)) #computing value of U
    frobNormLst_svd = []
    counter = 1
    if not os.path.exists('svd_images'): #checking if a folder exists with same name
        os.mkdir('svd_images')  #making a folder to save images for all values of k
    if not os.path.exists('ErrorImages_SVD'): #checking if a folder exists with same name
        os.mkdir('ErrorImages_SVD')  #making a folder to save images for all values of k 
    sigMatrix = np.zeros((256, 256)) #initialising a null matrix
    for i in range(256):
        fileName = 'svd_images/'+str(i+1)+'.jpg' #Setting a filename for new image
        sigMatrix[i][i] = sigma[i]
        svd_recon = U@sigMatrix@(V.T) #reconstructed image for current value of k
        tempFrobNorm = np.linalg.norm(imgMatrix-svd_recon) #computing Frob Norm
        frobNormLst_svd.append(tempFrobNorm)    #appending Frob Norm to list
        y = np.rint(np.absolute(svd_recon))
        tempImg = img.fromarray(y.astype(np.uint8)) #converting matrix to image
        if i+1 == counter:  #plotting reconstructed img vs error image
            pyplot.subplot(1, 2, 1)
            pyplot.imshow(tempImg, cmap="gray")
            plotName = "K = " + str(counter) + "\nReconstructed Image"
            pyplot.title(plotName)
            errorImg = np.rint(np.absolute(imgMatrix-svd_recon))
            tempErrImg = img.fromarray(errorImg.astype(np.uint8))
            pyplot.subplot(1, 2, 2)
            pyplot.imshow(tempErrImg, cmap="gray")
            plotName = "K = " + str(counter) + "\nError Image"
            pyplot.title(plotName)
            errfileName = "ErrorImages_SVD/"+str(i+1)+".png"
            if os.path.exists(errfileName):
                os.remove(errfileName)
            pyplot.savefig(errfileName)
            pyplot.close()
            counter = counter*2
        if os.path.exists(fileName):
            os.remove(fileName)
        tempImg.save(fileName)  #saving the image for current value of k
    pyplot.plot(frobNormLst_svd)    #plotting frobnorm vs k
    pyplot.xlabel('K-Value')
    pyplot.ylabel('Frobenius Norm')
    pyplot.title('Frobenius Norm vs K, SVD')
    if(os.path.exists('svd_graph.png')):
        os.remove('svd_graph.png')
    pyplot.savefig('svd_graph.png') #saving the graph
    pyplot.close()  #closing the plotted graph

def main():
    imgMatrix = np.asarray(img.open('39.jpg')).astype(np.int64) #converting img to matrix
    EVD(imgMatrix)  #Computing the Eigen Value decomposition of the Matrix 
    SVD(imgMatrix)  #Computing the Singular Value decomposition of the Matrix

main()
#print(eigVals) #print(eigVecs) #print(imgMatrix) #DebuggingStatements
