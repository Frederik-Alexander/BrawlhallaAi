import os
import json
import sys
import numpy as np
import random
import cv2
train_flag = True
if train_flag:
    from NN import NeuralNetwork
    from Train_VAE import Train_VAE
    from Train_Autoencoder import Train_autoencoder

class image_dict:
    def __init__(self,path):
        new_dict = {}
        files = os.listdir(path+"/")
        bar = progressBar(30,"images processed",len(files))
        for im in files:
            if not im.endswith(".png"):
                print("\nFile: {} is not a png!".format(path+"/"+im))
                continue
            title = im.split(".png")[0]
            id,action = title.split(" ",1)
            id = int(id)
            action = json.loads(action)
            im_path = path+"/"+im
            img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            im_hash = hash(img.tostring())
            new_dict[id] = {'action':action,'path':im_path, 'hash':im_hash}
            bar.update()
        print("\n")
        self.root_dir = os.path.basename(path)
        self.dict = new_dict
        self.path = path
        self.check_action_Vectors()

    def sort(self):
        keys = sorted(list(self.dict.keys()))
        new_dict = {}
        for key in keys:
            new_dict[key] = self.dict[key]
        self.dict = new_dict

    def shuffle(self):
        keys = list(self.dict.keys())
        random.shuffle(keys)
        new_dict = {}
        for key in keys:
            new_dict[key] = self.dict[key]
        self.dict = new_dict

    def delete_dublicates(self):
        self.sort()
        removeIds = []
        last_hash = ""
        before = len(self.dict)
        for id,data in self.dict.items():
            if data['hash'] == last_hash:
                os.remove(data['path'])
                removeIds.append(id)
                continue
            last_hash = data['hash']
        for id in removeIds:
            del self.dict[id]
        deleted = len(removeIds)
        print("{}/{} images deleted. {}%\n".format(deleted,before,round(deleted/before*100,2)))

    def get_list(self):
        return [(id,data) for id,data in self.dict.items()]

    def delete_all(self):
        for file in os.listdir(self.path+"/"):
            os.remove(self.path+"/"+file)
        os.rmdir(self.path)
        print("Session '{}' deleted.\n".format(self.root_dir))
        self.dict = {}

    def archive(self,archive_path):
        if not os.path.exists(archive_path+"/"+self.root_dir):
            os.makedirs(archive_path+"/"+self.root_dir)
        for file in os.listdir(self.path+"/"):
            os.rename(self.path+"/"+file,archive_path+"/"+self.root_dir+"/"+file)
        os.rmdir(self.path)
        print("Session '{}' archived.\n".format(self.root_dir))
        self.dict = {}
    
    def archiveInOne(self,archive_path):
        fusedPath = archive_path+"/fused"
        if not os.path.exists(fusedPath):
            os.makedirs(fusedPath)
        files = os.listdir(fusedPath)
        newId = len(files)
        for file in os.listdir(self.path+"/"):
            title = file.split(".png")[0]
            id,action = title.split(" ",1)

            newFile = str(newId)+" "+action+".png"
            while os.path.exists(fusedPath+"/"+newFile):
                newId += 1
                newFile = str(newId)+" "+action+".png"
            os.rename(self.path+"/"+file, fusedPath+"/"+newFile)
            newId += 1
        # os.rmdir(self.path)
        print("Session '{}' archived.\n".format(self.root_dir))
        self.dict = {}

    def check_action_Vectors(self):
        removeIds = []
        for id,data in self.dict.items():
            if sum(data['action']) != 1:
                os.remove(data['path'])
                removeIds.append(id)
        for id in removeIds:
            del self.dict[id]
        deleted = len(removeIds)
        if deleted > 0:
            print("{} images with incorrect action vector deleted\n".format(deleted))


class progressBar:
    def __init__(self,width,text,max):
        self.width = width
        self.max = max
        self.text = text
        self.counter = 0
        self.update()

    def update(self,num=-1):
        if num == -1:
            num = self.counter
            self.counter += 1
        percent = num/self.max
        bar = ('#'*int(percent*self.width)) + '-'*(self.width-int(percent*self.width))
        sys.stdout.write("\r[ {} ] {}/{} {text}".format(bar,num,self.max,text=self.text))
        sys.stdout.flush()


class data_manager:
    def __init__(self,data_list):
        self.list = data_list

    def stack(self,stack_size):
        result = []
        stacks = int(len(self.list)/stack_size)
        for i in range(stacks):
            result.append(self.list[i:i+stack_size])
        self.stacks = result

    def shuffle(self):
        random.shuffle(self.stacks)

    def batch(self,batch_size):
        result = []
        batch = []
        for stack in self.stacks:
            batch.append(stack)
            if len(batch) == batch_size:
                result.append(batch)
                batch = []
        result.append(batch)
        self.batches = result
    
    def fromBatch(self,batchnr,referenceImg, Subtract = True):
        batch = self.batches[batchnr]
        X_train = []
        Y_train = []
        for stack in batch:
            x = []
            y = []
            for id,data in stack:
                img = cv2.imread(data['path'],cv2.IMREAD_GRAYSCALE)
                img = self.finalprocessing(img,referenceImg) # freddy
                action = data['action']
                x.append(img)
                y.append(action)
            # format image stack
            x = np.asarray(x).transpose(2, 1, 0)
            X_train.append(x)
            Y_train.append(y)
        return (X_train, Y_train, stack_size)

    def finalprocessing(self,img,referenceImg):

        #state_size = (100,100)

        img = cv2.resize(img, input_shape_autoencoder_to_reshape) 
        referenceImg = cv2.resize(referenceImg, input_shape_autoencoder_to_reshape) 
        #img = np.resize(img,state_size)

        #img = np.resize(img,(input_shape_VAE[0],input_shape_VAE[1]))

        
        #referenceImg = np.resize( referenceImg,(input_shape_VAE[0],input_shape_VAE[1]))
        #img = self.generateSubtractedImages(img, referenceImg)
        return img 



    def generateSubtractedImages(self, img, referenceImg):

        image2 = img
        finalimg =   image2 -referenceImg

        
        """        thresh = 1
        im_bw = cv2.threshold(finalimg, thresh, 255, cv2.THRESH_BINARY)[1]
        """
        return finalimg

    def ToblackWhite(self,img):
        return img 


def train(training_data, trainClassVAE, trainClassAutoencoder ): ######################################TRAIN HERE#######################
    #trainClassVAE.fit(training_data)
    trainClassAutoencoder.fit(training_data)
    
path = os.path.dirname(__file__)
data_path = path+"/Data"

batch_size = 100
stack_size = 1
input_shape_autoencoder_to_reshape = (300,300)
input_shape_autoencoder = (300,300,1)
input_shape_VAE_inputdata = (800,800)
input_shape_VAE = (800,800) # resize to this value

archive_path = path+"/Data/archive"


referenceImg = cv2.imread(path + '/Reference Images/EMPTYMAP.png',0) # this is for subtracting

archive_path = path+"/Data/archive"
fusedPath_fredy = path+"/Data/fused"

if train_flag:
    trainClassAutoencoder = Train_autoencoder(input_shape_autoencoder,0.1)
    trainClassVAE = Train_VAE(input_shape_VAE,0)





for session in os.listdir(data_path):
    session_path = path+"/Data/"+session

    if session_path == archive_path or session_path == fusedPath_fredy  :
        # ignore archive folder
        continue
    print("Working on session: {}".format(session))

    if len(os.listdir(session_path+"/")) == 0:
        # delete empty folders
        os.rmdir(session_path)
        continue

    # initialize files
    print("Initializing session files...")
    images = image_dict(session_path)
    print("Deleting dublicate images...")
    images.delete_dublicates()
    # images.archiveInOne(path+"/Data/")

    if not train_flag:
        images.archive(archive_path)
        continue
    data = data_manager(images.get_list())
    data.stack(stack_size)
    data.shuffle()
    data.batch(batch_size)



    print("Training...")

    
    bar = progressBar(30,"batches",len(data.batches))
    for i in range(len(data.batches)):
        train(data.fromBatch(i,referenceImg,True),trainClassVAE, trainClassAutoencoder)
        bar.update()
    
    
    
    
    
    print("\nDone training.")
    images.archive(archive_path)

print("All sessions processed!")
