#!/usr/local/bin/python3


 # This file is part of QuickMatch.
 #
 # Copyright (C) 2018 Zachary Serlin <zserlin@bu.edu> (Boston University)
 # For more information see <https://bitbucket.org/tronroberto>
 #
 # QuickMatch is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # QuickMatch is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with QuickMatch. If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
import numpy as np
import cv2
from scipy.spatial import distance as scipydist
import csv
import pandas as pd
import random
from datetime import datetime
from PIL import Image
import glob
import os.path as path
import sys
np.set_printoptions(threshold=np.nan)


###############################################################################
#This class definition creates a struct data structure
class empty_struct:
    pass

###############################################################################
#This is the main function that calls all of the subfunctions of QuickShift.
#The main options in this function are the threshold and feature extraction method
def main(starttime):
    #Initialization of Important Constants and Features data structure
    threshold = .75
    method = 'SIFT' #Can use 'SIFT','ORB', or 'SURF'

    #Choose file path for main images
    dirname = path.abspath(path.join(__file__,"../.."))
    images_file_path = dirname+'/test_images/*.%s'

    #Build main data structure
    features = empty_struct()

    #Import Data into [dxn] numpy array
    features = get_imgs_data(features,method,images_file_path)
    features.num_img = len(features.images)

    sift_time = datetime.now() - starttime

    #This is the length and dimension of the feature vector
    features.size = features.data.shape
    print('Number and Size of Features')
    print(features.size)

#### Start QuickShift

    #Find Euclidean distance betweeen all features
    features.dist = distance(features.data, features.size)

    #Calculate Density from distance values matrix
    features = cal_density(features)

    #Build tree based on density
    features = build_kdtree(features)

    #Sort parent edges by length for cut and merge
    features = sort_edge_index(features)

    #Break tree where parent edges are larger than fixed threshold or in same image
    features = break_merge_tree(features, threshold)

#### End QuickShift

    print('Quickmatch Runtime: ')
    quickmatch_time = datetime.now() - startTime - sift_time
    print(quickmatch_time)

    print('Number or Features')
    print(features.size[0])

    print('Number of Clusters:')
    print(features.clusters.shape[0])

### Convert QuickMatch Matches into DMatch Format for reference image
    query_idx = 0
    #features = count_QM_matches(features,query_idx)

    #Show Matched Keypoints Sample
    print('Press Any Key to Show Next Image Set')
    for i in range(0,len(features.images)-1):
        print('Image Index Comparison')
        features_to_DMatch(features,query_idx,i)

    #Choose a random largest cluster to display
    (values,counts) = np.unique(features.cluster_member,return_counts=True)
    ind=np.where(counts == counts.max())
    rand_cluster= values[ind[0][np.random.randint(ind[0].shape[0])]]

    print('Press Any Key to Continue')
    print('Showing Largest Cluster')
    show_points(features,rand_cluster)

###############################################################################
#Import car images and run Sift to get dataset
def get_imgs_data(features,method,path):

    #Initalize data structures
    if method == 'SIFT':
        features.data = np.empty((1,128),dtype=np.float64)
        sift = cv2.xfeatures2d.SIFT_create(500,3,.1,5,1.6)
    if method == 'ORB':
        features.data = np.empty((1,32),dtype=np.float64)
        orb =cv2.ORB_create()
    if method == 'SURF':
        features.data = np.empty((1,64),dtype=np.float64)
        surf = cv2.xfeatures2d.SURF_create(400)

    first = 0
    features.keypoints = np.empty((1,2),dtype=object)
    features.member = np.empty((1),dtype=int)

    #Get raw image files
    image_list = [cv2.imread(item) for i in [sorted(glob.glob(path % ext)) for ext in ["jpg","gif","png","tga"]] for item in i]

    features.images = image_list

    #For each image, extract sift features and organize them into right structures
    for i in range(0,len(features.images)):
        gray= cv2.cvtColor(features.images[i],cv2.COLOR_BGR2GRAY)

        if method == 'SURF':
            kp, des = surf.detectAndCompute(gray,None)
        if method == 'SIFT':
            kp, des = sift.detectAndCompute(gray,None)
        if method == 'ORB':
            kp, des = orb.detectAndCompute(gray,None)
        keypts = [p.pt for p in kp]
        member = np.ones(len(keypts)) * i
        #add features to data structures
        if first > 0:
            features.kpts = np.hstack((features.kpts,kp))
            features.desc = np.vstack((features.desc,des))
            features.keypoints = np.concatenate((features.keypoints, keypts), axis=0)
            features.data = np.concatenate((features.data, des), axis=0)
            features.member = np.concatenate((features.member, member), axis=0)
        if first == 0:
            features.kpts = kp
            features.desc = des
            features.keypoints = np.concatenate((features.keypoints, keypts), axis=0)
            features.data = np.concatenate((features.data, des), axis=0)
            features.member = np.concatenate((features.member, member), axis=0)
            first = 1

    #Delete first empty artifact from stucture def
    features.keypoints = np.delete(features.keypoints,0,axis=0)
    features.data = np.delete(features.data,0,axis=0)
    features.data = np.random.normal(features.data,0.001)
    features.member = np.delete(features.member,0)

    return features

###############################################################################

#This Finds the n-dim euclidean dist. between points and returns the matrix of
#distances D
def distance(points,feature_length):
    D = scipydist.pdist(points,'euclidean')
    D = scipydist.squareform(D)
    return D

###############################################################################

#This function takes in the distance matrix
#and returns the density at each point.
#It also calculates the nearest distances between two features in the same image
#This becomes the bandwidth of the gaussian kernel used (times some constant).
###############################################################################
def cal_density(features):
    D = features.dist
    I = np.identity(D.shape[0]).astype(int)
    Dint = D.astype(int) - I
    D[Dint == -1] = np.nan

    num_feat = features.dist.shape[0]
    bandwidth = np.ones(num_feat)
    membership = features.member.astype(int)
    x = np.empty(features.num_img,dtype=object)

    #Loop through num images to find normalization for each one
    for i in range(0,features.num_img):
        x[i] = np.where(membership == i)
    for k in range(0,num_feat):
        first = x[membership[k]][0][0]
        last_idx = x[membership[k]][0].shape[0]
        last = x[membership[k]][0][last_idx-1]
        bandwidth[k] = np.nanmin(D[k,first:last])
    Density = np.zeros(num_feat)
    D[np.isnan(D)] = 0

    #Broadcast division of bandwidth
    D_corrected = D/bandwidth[:,None]

    #Calc Gaussian at each feature
    gaus = np.exp((-.5)*np.power(D_corrected,2))

    #Sum density values at each point
    Density = np.sum(gaus,axis=0)

    features.density = Density
    features.bandwidth = bandwidth

    return features

###############################################################################

#This function sort the parent edges from shortest to longest and returns the
#sorted indicies in the features.sorted_idx struct.
def sort_edge_index(features):
    features.sorted_idx = sorted(range(len(features.parent_edge)), key=lambda k: features.parent_edge[k])
    return features

###############################################################################


#This function takes in the feature density and determines the set of points
#that have a higher density and makes its nearest neighbor from that set
#its parent node. It then goes into its parent node(s) and makes itself a
#child of that node.

def build_kdtree(features):
#Initialize tree array or arrays
    features.parent = np.empty(features.size[0],dtype=object)
    features.parent_edge = np.empty(features.size[0],dtype=object)

#Build Tree starts here
    for i in range(0,features.size[0]):
        features.parent[i] = np.array([-1])
        features.parent_edge[i] = np.array([-1])
#Find larger density nodes here

        larger = np.transpose(np.nonzero(np.greater(features.density,features.density[i])))
#If the node is not the highest, find its parent
        if larger.shape[0] != 0:
            x = np.take(features.dist[i,:],larger)
            nearest = np.take(larger,(np.where(x == x.min())))
            dist_min = x.min()
            features.parent[i] = nearest[0]
            features.parent_edge[i] = dist_min
    return features


###############################################################################
#This function breaks edges longer than the threshold and assigned the broken
#Child to a new parent node

def break_merge_tree(features, threshold):

    features.cluster_member = np.arange(features.size[0])
    features.matchden = features.bandwidth

    for j in range(0,features.size[0]):
        idx = features.sorted_idx[j]
        parent_idx = features.parent[idx][0]

        if parent_idx != -1:
            min_dens = np.minimum(features.matchden[idx],features.matchden[parent_idx])
            x = np.take(features.member,np.where(features.cluster_member == features.cluster_member[parent_idx]))
            y = np.take(features.member,np.where(features.cluster_member == features.cluster_member[idx]))
            isin_truth = np.isin(x,y)

            #Only consider points that meet criteria
            if (features.parent_edge[idx] < (threshold*min_dens)) and not(isin_truth.any()):
                features.cluster_member[features.cluster_member == features.cluster_member[idx]] = features.cluster_member[parent_idx]
                features.matchden[features.cluster_member == features.cluster_member[idx]] = min_dens
                features.matchden[features.cluster_member == features.cluster_member[parent_idx]] = min_dens

        (values,counts) = np.unique(features.cluster_member,return_counts=True)
        features.clusters = counts
    return features

###############################################################################
#End of QuickMatch
###############################################################################


#This function turns each match from QuickMatch into the OpenCV DMatch Structure
#The matches are saved to features.Matches_QM
def count_QM_matches(features,query_idx):
    im_idx1 = query_idx
    x = np.take(features.cluster_member,np.where(features.member == im_idx1))
    for i in range(0,len(features.images)):
        y = np.take(features.cluster_member,np.where(features.member == i))
        match = np.intersect1d(x,y)
        first = 0
        im_idx2 = i
        if match.shape[0] == 0:
            a = 0
        else:
            for j in range(0,match.shape[0]):
                fet1_idxa = np.where(features.cluster_member == match[j])
                fet1_idxb = np.where(features.member == im_idx1)
                fet1_idx = np.intersect1d(fet1_idxa,fet1_idxb)
                fet2_idxa = np.where(features.cluster_member == match[j])
                fet2_idxb = np.where(features.member == im_idx2)
                fet2_idx = np.intersect1d(fet2_idxa,fet2_idxb)
                desc_dist = features.dist[fet1_idx,fet2_idx]
                dpoint = cv2.DMatch(fet1_idx,fet2_idx,im_idx1,desc_dist)
                if first == 0:
                    DMatches = dpoint
                    first = 1
                if first > 0:
                    DMatches = np.hstack((DMatches,dpoint))
            features.Matches_QM[i] = DMatches

    return(features)

###############################################################################
#Convert sets of images to DMatch stucture

def features_to_DMatch(features,im_idx1,im_idx2):
    x = np.take(features.cluster_member,np.where(features.member == im_idx1))
    y = np.take(features.cluster_member,np.where(features.member == im_idx2))
    match = np.intersect1d(x,y)
    first = 0
    image1 = features.images[im_idx1].copy()
    image2 = features.images[im_idx2].copy()

    if match.shape[0] == 0:
        print('No Matches between those images')
        print(im_idx1,im_idx2)
        return()
    for i in range(0,match.shape[0]):
        fet1_idxa = np.where(features.cluster_member == match[i])
        fet1_idxb = np.where(features.member == im_idx1)
        fet1_idx = np.intersect1d(fet1_idxa,fet1_idxb)
        fet2_idxa = np.where(features.cluster_member == match[i])
        fet2_idxb = np.where(features.member == im_idx2)
        fet2_idx = np.intersect1d(fet2_idxa,fet2_idxb)
        desc_dist = features.dist[fet1_idx,fet2_idx]
        dpoint = cv2.DMatch(fet1_idx,fet2_idx,im_idx1,desc_dist)
        if first == 0:
            DMatches = dpoint
            first = 1
        if first > 0:
            DMatches = np.hstack((DMatches,dpoint))
    print(im_idx1,im_idx2)
    image1 = features.images[im_idx1].copy()
    image2 = features.images[im_idx2].copy()
    img3 = cv2.drawMatches(image1,features.kpts,image2,features.kpts,DMatches,1)

    cv2.imshow("Image",img3)
    cv2.waitKey()
    return(DMatches)


###############################################################################
#Draw image patches for clusters
###############################################################################
def show_keypoint_patch(features,rand_cluster):

    kp_idx = np.transpose(np.nonzero(np.equal(features.cluster_member,features.cluster_member[rand_cluster])))
    patch = 25
    imed = 0
    for i in range(0,kp_idx.shape[0]):
        kp = features.keypoints[kp_idx[i]].astype(int)

        x = int(features.member[kp_idx[i][0]])
        y = int(kp[0][0]-patch)
        z = int(kp[0][0]+patch)
        q = int(kp[0][1]-patch)
        p = int(kp[0][1]+patch)
        if y>0 and q >0 and z < features.images[x].shape[0] and p < features.images[x].shape[1]:
            im_patch = features.images[x][y:z,q:p]
            if imed == 0:
                im = im_patch
                imed = 1
            else:
                im = np.concatenate((im,im_patch),axis=1)
    if imed > 0:
        cv2.imshow("Image",im)
        cv2.waitKey()


###############################################################################
#Draw keypoints in images for a cluster
###############################################################################
def show_points(features,rand_cluster):
    kp_idx = np.transpose(np.nonzero(np.equal(features.cluster_member,features.cluster_member[rand_cluster])))
    patch = 10
    imed = 0

    for i in range(0,kp_idx.shape[0]):
        base_im_idx = features.member[kp_idx[i]]
        kp = features.keypoints[kp_idx[i]].astype(int)
        a = int(kp[0][0])
        b = int(kp[0][1])
        x = int(features.member[kp_idx[i][0]])
        im_proc = features.images[x].copy()
        y = int(kp[0][0]-patch)
        z = int(kp[0][0]+patch)
        q = int(kp[0][1]-patch)
        p = int(kp[0][1]+patch)
        if y>-1 and q >-1 and z < im_proc.shape[0] and p < im_proc.shape[1]:
            im_patch = cv2.circle(im_proc,(a,b),patch,(0,255,0),-1)
            if imed == 0:
                dim = (400, 300)
                im_patch = cv2.resize(im_patch, dim, interpolation = cv2.INTER_AREA)
                im = im_patch
                imed = 1
            else:
                dim = (400, 300)
                im_patch = cv2.resize(im_patch, dim, interpolation = cv2.INTER_AREA)
                im = np.concatenate((im,im_patch),axis=1)

    #To resize images smaller, uncomment below:
    #r = 100.0 / im.shape[1]
    #dim = (1800, 350)
    #im = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
    #cv2.imwrite('Clustering4.jpg',im)

    if imed >0:
        cv2.imshow("Image",im)
        cv2.waitKey()
        cv2.destroyWindow("Image")

###############################################################################
#This runs the main function and tracks the total execution time
if __name__== "__main__":
    startTime = datetime.now()
    main(startTime)
