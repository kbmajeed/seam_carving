
"""
Seam Carving for Content-Aware Image Resizing
(c)Abdulmajeed Muhammad Kabir
"""



"""
Dependencies
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



"""
Functions
"""
def load_image_gray(path):
    img   = cv2.imread(path)
    img   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
   
def load_image_colr(path):
    img   = cv2.imread(path)
    return img

def energy_map(img):
    imgdx  = cv2.Sobel(img, cv2.CV_64F,1,0)  
    imgdy  = cv2.Sobel(img, cv2.CV_64F,0,1)  
    imgMap = abs(imgdx) + abs(imgdy)
    return imgMap

def energy_map2(img):
    imgdx  = cv2.Sobel(img, cv2.CV_64F,1,0)  
    imgdy  = cv2.Sobel(img, cv2.CV_64F,0,1)  
    imgMap = np.sqrt((imgdx)**2 + (imgdy)**2)
    return imgMap

def compute_cost_map(energy_map):
    imgTemp1   = energy_map.copy()
    imgCumMinE = np.zeros_like(imgTemp1)
    r_imgTemp1 = imgTemp1.shape[0]
    c_imgTemp1 = imgTemp1.shape[1]

    imgCumMinE[0,:] = imgTemp1[0,:]
    for row in range(1, r_imgTemp1):
        for col in range(0, c_imgTemp1):
            if col == 0:
                imgCumMinE[row,col] = imgTemp1[row,col] + np.min(np.array ((imgTemp1[row-1,col],imgTemp1[row-1,col+1])))
            elif col == c_imgTemp1-1:
                imgCumMinE[row,col] = imgTemp1[row,col] + np.min(np.array ((imgTemp1[row-1,col-1],imgTemp1[row-1,col])))
            else:
                imgCumMinE[row,col] = imgTemp1[row,col] + np.min(np.array ((imgTemp1[row-1,col-1],imgTemp1[row-1,col],imgTemp1[row-1,col+1])))
    return imgCumMinE

def find_verticalSeam(imgCumMinE):
    imgBackTrack   = imgCumMinE.copy()
    r_imgBackTrack = imgBackTrack.shape[0]
    c_imgBackTrack = imgBackTrack.shape[1]

    argRowMin         = int(np.argmin(imgBackTrack[-1, :]))
    argRowMinList     = np.zeros(r_imgBackTrack)
    argRowMinList[-1] = argRowMin
  
    argRowMinListMap = np.zeros((r_imgBackTrack,c_imgBackTrack))
    argRowMinListMap[-1, argRowMin]

    for row in range(-1, -r_imgBackTrack, -1):
        if argRowMin == 0:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin],imgBackTrack[row-1,argRowMin+1])))
            argRowMin = int((argRowMinNext) + (argRowMinList[row])) ##watch bug
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
        elif argRowMin == c_imgBackTrack-1:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin-1],imgBackTrack[row-1,argRowMin])))
            argRowMin = int((argRowMinNext) + (argRowMinList[row])) -1  ##possible issue
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
        else:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin-1],imgBackTrack[row-1,argRowMin],imgBackTrack[row-1,argRowMin+1])))
            argRowMin = int((argRowMinNext - 1) + (argRowMinList[row]))
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
    return argRowMinList#, argRowMinListMap

def remove_verticalSeam(img, argRowMinList):
    r_imgBackTrack = img.shape[0]
    c_imgBackTrack = img.shape[1]
    imgTarget = np.zeros((r_imgBackTrack,c_imgBackTrack-1))
    for row in range(0, -r_imgBackTrack, -1):
        if argRowMinList[row-1] == 0:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
        elif argRowMinList[row-1] == c_imgBackTrack-1:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
        else:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
    return imgTarget

def find_horizontalSeam(imgCumMinE):
    imgBackTrack   = imgCumMinE.copy()
    imgBackTrack   = np.rot90(imgBackTrack,1,axes=(1,0))
    r_imgBackTrack = imgBackTrack.shape[0]
    c_imgBackTrack = imgBackTrack.shape[1]

    argRowMin         = int(np.argmin(imgBackTrack[-1, :]))
    argRowMinList     = np.zeros(r_imgBackTrack)
    argRowMinList[-1] = argRowMin

    argRowMinListMap = np.zeros((r_imgBackTrack,c_imgBackTrack))
    argRowMinListMap[-1, argRowMin]

    for row in range(-1, -r_imgBackTrack, -1):
        if argRowMin == 0:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin],imgBackTrack[row-1,argRowMin+1])))
            argRowMin = int((argRowMinNext) + (argRowMinList[row])) ##watch bug
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
        elif argRowMin == c_imgBackTrack-1:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin-1],imgBackTrack[row-1,argRowMin])))
            argRowMin = int((argRowMinNext) + (argRowMinList[row])) -1  ##possible issue(now fixed)
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
        else:
            argRowMinNext = np.argmin(np.array ((imgBackTrack[row-1,argRowMin-1],imgBackTrack[row-1,argRowMin],imgBackTrack[row-1,argRowMin+1])))
            argRowMin = int((argRowMinNext - 1) + (argRowMinList[row]))
            argRowMinList[row-1] = argRowMin
            argRowMinListMap[row-1, argRowMin] = 1
    return argRowMinList#, argRowMinListMap

def remove_horizontalSeam(img, argRowMinList):
    img = np.rot90(img,1,axes=(1,0))
    r_imgBackTrack = img.shape[0]
    c_imgBackTrack = img.shape[1]
    imgTarget = np.zeros((r_imgBackTrack,c_imgBackTrack-1))
    for row in range(0, -r_imgBackTrack, -1):
        if argRowMinList[row-1] == 0:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
        elif argRowMinList[row-1] == c_imgBackTrack-1:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
        else:
            argRowMin = int(argRowMinList[row-1])
            imgTarget[row-1,:] = np.concatenate([img[row-1,:argRowMin],img[row-1,argRowMin+1:]])
    return np.rot90(imgTarget,1,axes=(0,1))

def reduceWidth(img, no_Cols=10):
    imgNewWidth = img.copy()
    pbar = tqdm(total=no_Cols)
    for i in range(no_Cols):
        imgNewWidthE = energy_map(imgNewWidth)
        imgNewWidthC = compute_cost_map(imgNewWidthE)
        verticalSeam = find_verticalSeam(imgNewWidthC)
        imgNewWidth = remove_verticalSeam(imgNewWidth, verticalSeam)
        pbar.update()#(no_Cols/10)
    pbar.close()
    return imgNewWidth                                                                                                          

def reduceWidthColor(img_gray, img_colr, N):
    imgFinal = img_colr.copy() 
    for i in range(N):
        img_enrg = energy_map(img_gray)
        img_cost = compute_cost_map(img_enrg)
        img_vseam = find_verticalSeam(img_cost)
        imgFinalb = remove_verticalSeam(imgFinal[:,:,0], img_vseam)
        imgFinalg = remove_verticalSeam(imgFinal[:,:,1], img_vseam)
        imgFinalr = remove_verticalSeam(imgFinal[:,:,2], img_vseam)
        imgFinal = np.dstack((imgFinalb, imgFinalg, imgFinalr))
    return imgFinal

def reduceHeight(img, no_Rows=10):
    imgNewHeight = img.copy()
    pbar = tqdm(total=no_Rows)
    for i in range(no_Rows):
        imgNewHeightE = energy_map(imgNewHeight)
        imgNewHeightC = compute_cost_map(imgNewHeightE)
        horizontalSeam = find_horizontalSeam(imgNewHeightC)
        imgNewHeight = remove_horizontalSeam(imgNewHeight, horizontalSeam)
        pbar.update()#(no_Rows/10)
    pbar.close()
    return imgNewHeight

def find_N_verticalSeams(img, N):
    imgEnergyMap = energy_map(img)
    imgCostMap = compute_cost_map(imgEnergyMap)
    imgCumMinE = imgCostMap
    
    verticalSeamMap = np.zeros((img.shape[0],N))
    for i in range(N):
        imgEnergyMap = energy_map(img)
        imgCostMap = compute_cost_map(imgEnergyMap)
        imgCumMinE = imgCostMap
        verticalSeamMap[:,i] = find_verticalSeam(imgCumMinE)
        img = remove_verticalSeam(img, verticalSeamMap[:,i])
    return verticalSeamMap

def find_N_horizontalSeams(img, N):
    imgEnergyMap = energy_map(img)
    imgCostMap = compute_cost_map(imgEnergyMap)
    imgCumMinE = imgCostMap
    
    horizontalSeamMap = np.zeros((img.shape[1], N))
    for i in range(N):
        imgEnergyMap = energy_map(img)
        imgCostMap = compute_cost_map(imgEnergyMap)
        imgCumMinE = imgCostMap
        horizontalSeamMap[:,i] = find_horizontalSeam(imgCumMinE)
        img = remove_horizontalSeam(img, horizontalSeamMap[:,i])
    return horizontalSeamMap

def plotVSeam(img, verticalSeam):
    #seam is a single array with [n,:]
    img2 = img.copy()
    r_img = img2.shape[0]
    for row in range(r_img):
        col = int(verticalSeam[row])
        img2[row,col] = 255
    plt.imshow(img2, cmap='jet')

def plotHSeam(img, hortizontalSeam):
    #seam is a single array with [:,n]
    img2 = img.copy()
    c_img = img2.shape[1]
    for col in range(c_img):
        row = int(hortizontalSeam[col])
        img2[row,col] = 255
    plt.imshow(img2, cmap='jet')    

def create_newVPixels(img, vSeam):
    seam = vSeam.copy()
    verticalPixels = np.zeros(len(seam))
    for rows in range(img.shape[0]):
        verticalPixels[rows] = (float(img[rows, int(seam[rows]-1)]) + float(img[rows, int(seam[rows]+1)])) / 2
    return verticalPixels  

def create_newHPixels(img, hSeam): #incomplete
    seam = hSeam.copy()
    horizontalPixels = np.zeros(len(seam))
    for cols in range(img.shape[1]):
        horizontalPixels[cols] = (float(img[int(seam[cols]-1), cols]) + float(img[int(seam[cols]+1), cols])) / 2
    return horizontalPixels
    
def insert_newVPixels(img, seam, location='left'):
    verticalPixels = create_newVPixels(img, seam)
    if location == 'left':
        imgTarget = np.zeros((img.shape[0],img.shape[1]+1))
        for row in range(len(seam)):
            imgTarget[row,:] = np.hstack([img[row,:int(seam[row])],verticalPixels[row],img[row,int(seam[row]):]]) #insert to left of seam
    elif location == 'right':
        imgTarget = np.zeros((img.shape[0],img.shape[1]+1))
        for row in range(len(seam)):
            imgTarget[row,:] = np.hstack([img[row,:int(seam[row]+1)],verticalPixels[row],img[row,int(seam[row]+1):]]) #insert to right of seam
    elif location == 'both':        
        imgTarget = np.zeros((img.shape[0],img.shape[1]+2))
        for row in range(len(seam)):
            imgTarget[row,:] = np.hstack([img[row,:int(seam[row])],verticalPixels[row],img[row,int(seam[row])],verticalPixels[row],img[row,int(seam[row]+1):]]) #insert double seam
    else:
        imgTarget = imgTarget
    return imgTarget

def insert_newHPixels(img, seam, location='up'):  #incomplete
    horizontalPixels = create_newHPixels(img, seam)
    if location == 'up':
        imgTarget = np.zeros((img.shape[0]+1,img.shape[1]))
        for col in range(len(seam)):
            imgTarget[col,:] = np.hstack([img[:int(seam[col]),col],horizontalPixels[col],img[int(seam[col]):,col]]) #insert to up of seam
    elif location == 'down':
        imgTarget = np.zeros((img.shape[0]+1,img.shape[1]))
        for col in range(len(seam)):
            imgTarget[col,:] = np.hstack([img[:int(seam[col]+1),col],horizontalPixels[col],img[int(seam[col]+1):,col]]) #insert to up of seam            
    elif location == 'both':        
        imgTarget = np.zeros((img.shape[0]+2,img.shape[1]))
        for col in range(len(seam)):
            imgTarget[col,:] = np.hstack([img[:int(seam[col]),col],horizontalPixels[col],img[col,int(seam[col])],horizontalPixels[col],img[col,int(seam[col]+1):]]) #insert double seam
    else:
        imgTarget = imgTarget
    return imgTarget
    
def expandWidth(img, N, location='left'):
    imgTarget = img.copy()
    seams = find_N_verticalSeams(imgTarget,N)
    for i in range(N):
        seam = seams[:,i]
        imgTarget = insert_newVPixels(imgTarget, seam, location)
    return imgTarget

def expandHeight(img, N, location='up'):  #incomplete
    imgTarget = img.copy()
    seams = find_N_horizontalSeams(imgTarget,N)
    for i in range(N):
        seam = seams[:,i]
        imgTarget = insert_newHPixels(imgTarget, seam, location)
    return imgTarget

def expandWidthColor(img_gray, img_colr, N):
    imgFinal = img_colr.copy() 
    if N >= imgFinal.shape[1]:
        N = N-1    
    imgFinal_vseam = find_N_verticalSeams(img_gray, N)
    for i in range(N):
        imgFinalb_new = insert_newVPixels(imgFinal[:,:,0], imgFinal_vseam[:,i], location='left')
        imgFinalg_new = insert_newVPixels(imgFinal[:,:,1], imgFinal_vseam[:,i], location='left')
        imgFinalr_new = insert_newVPixels(imgFinal[:,:,2], imgFinal_vseam[:,i], location='left')
        imgFinal  = np.dstack((imgFinalb_new , imgFinalg_new , imgFinalr_new ))
    imgFinal_expand = imgFinal.astype('uint8')
    return imgFinal_expand


def seamVEnergy(energyMap, seam):
    #seam = single array of seam
    img = energyMap.copy()
    seamEnergy = 0
    for rows in range(img.shape[0]):
        seamEnergy += float(img[rows, int(seam[rows])])
    return seamEnergy

def seamHEnergy(energyMap, seam):
    #seam = single array of seam
    img = energyMap.copy()
    seamEnergy = 0
    for rows in range(img.shape[1]):
        seamEnergy += float(img[int(seam[rows]), rows])
    return seamEnergy

def showSeamsColor(img_gray, img_colr, N):
    ### Visualizing the Seams in the order of removal # PLOT N Seams ###
    #N = int((50./100)*(img_gray.shape[1])) #Enlarge by 50%
    img = img_gray.copy()
    verticalSeam = find_N_verticalSeams(img, N)
    verticalSeam = verticalSeam.astype('int')
    for i in range(N):
        fnvs = verticalSeam[:,i]
        for row in range(img.shape[0]):
            col = int(fnvs[row])
            img[row,col] = 255
        #plt.imshow(img, cmap='jet')
    name = 'window'; cv2.namedWindow(name); cv2.moveWindow(name, 500,300)
    cv2.imshow(name, img.astype('uint8')); cv2.waitKey(); cv2.destroyAllWindows()
    cv2.imwrite("ImgSeamMap.png", img)

def transportMap(img, Nr=5, Nc=5):
    #image = img.copy()
    mapping = np.zeros((Nr+Nc))
    efvs = 0
    efhs = 0
    for i in range(Nr+Nc):    
        energyMap = energy_map(img)
        costMap = compute_cost_map(energyMap)
        
        fvs = find_verticalSeam(costMap)
        fhs = find_horizontalSeam(costMap)
        
        efvs = seamVEnergy(energyMap,fvs) + efvs#[-1]
        print efvs
        efhs = seamHEnergy(energyMap,fhs) + efhs#[-1]
        print efhs

        if efvs < efhs:
            print('delete column')
            img = remove_verticalSeam(img, fvs)
            mapping[i] = 1
            
        elif efhs < efvs:
            print('delete row')
            img = remove_horizontalSeam(img, fhs)
            mapping[i] = -1
            
        else:
            print('doing nothing')
            img = img
            mapping[i] = 0
            
    return mapping #transportMap.astype('float64')
          
#    transportMap = np.zeros_like(img) #initializeTransportMap
#       
#    for i in range(0, Nc):
#        for j in range(0, Nr):
#            #minE = np.argmin((envs[i], enhs[j]))
#            minE = np.argmin((envs[i+1]+envs[i], enhs[j+1]+enhs[j]))
#            transportMap[i+1,j+1] = minE
#    return transportMap





