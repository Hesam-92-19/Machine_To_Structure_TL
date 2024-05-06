import shutil
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
from IPython.display import HTML
import ipywidgets as widgets
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class RM_FS_TL():
    """
    The following initialization first download and extract required files to start the analysis
    These files include: Block geometries, Google (post-event) and Bing (pre-event) images
    which were taken from Lahaina Maui fire incident herein.
    
    Parameters:
        width (int): Width of images in pixels
        height (int): Height of images in pixels
        
    Returns:
        Build ./Data folder with the aformentioned files within
    """
    def __init__(self,cwd):
    # Download required files and set the directory
        self.cwd=cwd
        path='./RM_FS_TL'
        if os.path.exists(path):
            pass
        else:
            url = 'https://www.dropbox.com/scl/fi/438cfug8ks6ye05zly4c4/RM_FS_TL.zip?rlkey=iiuadbrbeukf29t1goo7nyt75&st=2v0315ph&dl=1'
            destination = cwd+'Files.zip'
            import urllib
            #Some functions to handle downloads
            def download_file(url, destination):
                urllib.request.urlretrieve(url, destination)

            def is_download_complete(url, destination):
                # Get the size of the file from the Content-Length header
                with urllib.request.urlopen(url) as response:
                    expected_size = int(response.headers['Content-Length'])

                # Get the actual size of the downloaded file
                actual_size = os.path.getsize(destination)

                # Compare the expected size with the actual size
                return expected_size == actual_size

            download_file(url, destination)

            if is_download_complete(url, destination):
                print("Download complete!")
            else:
                print("Download Failed; Please retry")

            # Building the file hierachy
            ''' The file hierachy is as follows

            │── RM-FS-TL
            │       └── Models
            │           └── Blocks_geom*.parquet
            │           └── Blocks_geom*.parquet
            │           └── Blocks_geom*.parquet
            │           └── A_RM__S4_1000.npy (Rot. Machinery dataset "A", i.e., torque 0 Nm; 4-channel dataset)
            │           └── A_RM_Classes_1000.npy (Rot. Machinery dataset "A", i.e., torque 0 Nm; number of samples in each class)
            │           └── B_RM__S4_1000.npy (Rot. Machinery dataset "B", i.e., torque 2 Nm; 4-channel dataset)
            │           └── B_RM_Classes_1000.npy (Rot. Machinery dataset "B", i.e., torque 2 Nm; number of samples in each class)
            │           └── C_RM__S4_1000.npy (Rot. Machinery dataset "C", i.e., torque 4 Nm 4-channel dataset)
            │           └── C_RM_Classes_1000.npy (Rot. Machinery dataset "C", i.e., torque 4 Nm; number of samples in each class)
            │           └── Yellow_S4_1000.npy (Yellow Frame 4-channel dataset)
            │           └── Yellow_Classes_1000.npy (Yellow Frame 4-channel dataset; number of samples in each class)
            │           └── QUGS_S4_1000.npy (QUGS Frame 4-channel dataset)
            │           └── QUGS_Classes_1000.npy (QUGS 4-channel dataset; number of samples in each class)
            '''

            zip_file_path = cwd+'Files.zip'
            extract_dir = cwd

            shutil.unpack_archive(zip_file_path, extract_dir)
            print(f"Zip file extracted to {extract_dir}")

    '''
    This function performs the Domain Adaptation routine of Algorithm 1 in the article, 

    Parameters:
        RM_N (str): RM dataset (source) name, ehrein it can be 'A', 'B' or 'C' 
        Target_N (str): Target frame dataset name, ehrein it can be 'Yellow', and 'QUGS'
        
    Returns:
        Transformed target F feature.

    '''
    def Algorith1(self,RM_N,Target_N,Coef=0.25,S=4,W=1000):
        cwd=self.cwd
        # Rotor data
        Rot_Data=np.load(cwd+'RM_FS_TL/'+RM_N+'_RM_S4_'+str(W)+'.npy')
        Class_L=int(np.floor(np.min(np.load(cwd+'RM_FS_TL/'+RM_N+'_RM_Classes_'+str(W)+'.npy'))/100))*100
        # Target Data 
        Yc=np.load(cwd+'RM_FS_TL/'+Target_N+'_Classes_'+str(W)+'.npy')
        Yc=np.concatenate(([0],Yc))
        # Cummulative class length
        E1=np.cumsum(Yc)
        Yel=np.load(cwd+'RM_FS_TL/'+Target_N+'_S'+str(S)+'_'+str(W)+'.npy')
        [A,B]=Yel.shape
        Yel=torch.tensor(Yel,device='cuda').reshape((A,S*int(W/2))).float()

        #Rot data training portion for DA mapping
        N=torch.tensor(Rot_Data[0:Class_L-100,:],device='cuda').reshape((Class_L-100,1,int(S*W/2))).float()
        N1=N.reshape((Class_L-100,int(W/2*S))).detach().cpu().numpy()
        for k in range(S):
            GA=np.mean(N1[:,k*int(W/2):(k+1)*int(W/2)]**2, axis=0)
            if k==0:
                G=GA*1
            else:
                G=np.concatenate((G,GA))
        GA=G
        
        ##########
        # FT, Algorith 1 --- Spectrum normalization using only the intact data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        E1Q=E1*1
        DMQ=Yel.detach().cpu().numpy()
        DMMCopy=np.copy(DMQ)
        for k in range(S):
            GA=G[k*int(W/2):(k+1)*int(W/2)];
            G1=np.mean(DMQ[0:int(E1[1]*Coef),k*int(W/2):(k+1)*int(W/2)]**2, axis=0)
            #G1=G1/np.max(G1)
            L=np.argsort(GA)
            L1=np.argsort(G1)
            Kok=np.zeros((int(W/2),int(W/2)))
            for j in range(int(W/2)):
                Kok[j,:]=(np.abs(GA-G1[j]))
            K=np.argmin(Kok, axis=1)

        # Transfroming the rest of dataset suing mapping from the intact portion %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i in range(int(W/2)):
                DMMCopy[:,L[i]+k*int(W/2)]=np.copy(DMQ[:,L1[i]+k*int(W/2)])*np.sqrt(GA[int(L[i])]/G1[int(L1[i])])
        [A,B]=DMMCopy.shape
        Ytel=torch.tensor(DMMCopy,device='cuda').reshape((A,1,int(S*W/2)))
        return Ytel, E1
    
    '''
    This function output the SDD outcome for a given target and source domain

    Parameters:
        RM_Class (str): RM dataset (source) class name: herein 1 to 9
        folder (str): Folder name of the saved models; 'PT' pre-trained models, 'NT' new models you train through "CL-Training.ipynb".
        RM_N (str): RM dataset (source) name, ehrein it can be 'A', 'B' or 'C' 
        
    Returns:
        nothing but a plot.

    '''
        
    def single_model(self,RM_Class,folder,RM_N,Target_N):
        cwd=self.cwd
        S=4
        W=1000
        Coef=0.25
        Net=torch.load(cwd+"RM_FS_TL/Models/"+folder+"/Model"+RM_N+str(RM_Class)+'_W_'+str(W)+".pth")
        [Ytel,E1]=self.Algorith1(RM_N,Target_N)
        FT=[]
        UU=np.zeros((len(Ytel),1))
        DD=Net.forward(Ytel).detach().cpu().numpy().reshape((1,-1))
        UU[:,0]=DD
        T=np.mean(UU[0:int(E1[1]*Coef),0])+3*np.std(UU[0:int(E1[1]*Coef),0])
        UU[:,0]=T-UU[:,0]
        UU[:,0][UU[:,0]>=0]=0
        UU[:,0][UU[:,0]<0]=1
        KOKO=np.sum(UU,axis=1) 
        KOKO[KOKO<=np.floor(1/2)]=0
        KOKO[KOKO>np.floor(1/2)]=1
        F1=[]
        for i in range(len(E1)-2):
            N1=KOKO[0:E1[1]]
            P1=KOKO[E1[i+1]:E1[i+2]]
            NN=len(N1)
            PP=len(P1)
            TN=NN-np.sum(N1)
            FP=np.sum(N1)
            FN=PP-np.sum(P1)
            TP=np.sum(P1)
            F1.append((2*TP)/(2*TP+FN+FP))
        plt.plot(DD.reshape(-1,1))
        #Define the two points
        #Generate x values for the line
        x = np.linspace(0, DD.shape[1], 100)
        y = x*0 + T
        
        #Create the plot
        plt.plot(x, y,'r--')

        current_ticks = plt.gca().get_yticks()
        
        #Add the special tick values/labels
        special_tick = T
        new_ticks = np.append(current_ticks, special_tick)
        plt.yticks(new_ticks)    
        plt.xlabel('Data Instance', fontsize=14, fontname='Times New Roman')
        plt.ylabel('$S$',fontsize=14, fontname='Times New Roman')
        plt.title('Example Plot', fontsize=18, fontname='Times New Roman')
        print('Mean F1 score across all classes equals:\t',np.mean(F1))
        plt.ylim([0,np.max(DD)])
        plt.xlim([0,(DD.shape[1])])
        

    '''
    This function output the SDD outcome for a given target and source domain

    Parameters:
        folder (str): Folder name of the saved models; 'PT' pre-trained models, 'NT' new models you train through "CL-Training.ipynb".
        RM_N (str): RM dataset (source) name, ehrein it can be 'A', 'B' or 'C' 
        Target_N (str): Target frame dataset name, ehrein it can be 'Yellow', and 'QUGS'
        
    Returns:
        nothing but a plot of ensembeling from 1 to 9 source classes of RM datasets.

    '''
        
    def ensemble_model(self,RM_N,Target_N,folder):
        Coef=0.25
        S=4
        W=1000
        cwd=self.cwd
        Rot_Data=np.load(cwd+'RM_FS_TL/'+RM_N+'_RM_S4_'+str(W)+'.npy')
        Class_N=np.cumsum(np.load(cwd+'RM_FS_TL/'+RM_N+'_RM_Classes_'+str(W)+'.npy'))
        Class_N = np.insert(Class_N, 0, 0)
        Rot_Data=Rot_Data[0:Class_N[15],:]
        Rot_Label=np.zeros((Rot_Data.shape[0],1))
        for i in range(15):
            Rot_Label[Class_N[i]:Class_N[i+1],0]=i
        #Apply 2D PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(Rot_Data)
        
        Means=np.zeros((15,4))
        for i in range(15):
            Means[i,:]=np.mean(pca_data[Class_N[i]:Class_N[i+1]:100,:])
        Means[:,2]=np.sqrt((Means[:,0]-Means[0,0])**2+(Means[:,1]-Means[0,1])**2)
        dist_max=np.max(Means[:,2])
        Means[:,3]=(Means[:,2]/dist_max)
        Weights=Means[1:10,3]*1
        Order=np.argsort(Weights)
        
        
        # Target Data 
        Yc=np.load(cwd+'RM_FS_TL/'+Target_N+'_Classes_'+str(W)+'.npy')
        Yc=np.concatenate(([0],Yc))
        E1=np.cumsum(Yc)
        Yel=np.load(cwd+'RM_FS_TL/'+Target_N+'_S'+str(S)+'_'+str(W)+'.npy')
        [A,B]=Yel.shape
        Yel=torch.tensor(Yel,device='cuda').reshape((A,S*int(W/2))).float()

        
        F1_Mean=[]
        Ensembles=np.arange(0,len(Order),2)
        #Ensembles=[0]
        [Ytel,E1]=self.Algorith1(RM_N,Target_N)
        for ie in Ensembles:
            Nets=[]
            for t in range(ie+1):
                RM_Class=Order[t]+1
                Nets.append(torch.load(cwd+"RM_FS_TL/Models/"+folder+"/Model"+RM_N+str(RM_Class)+'_W_'+str(W)+".pth"))
            UU=np.zeros((len(Yel),len(Nets)))
            i=0
            for Net in Nets:
                UU[:,i]=Net.forward(Ytel).detach().cpu().numpy().reshape((1,-1))
                i+=1
            for i in range(len(Nets)):   
                T=np.mean(UU[0:int(E1[1]*Coef),i])+3*np.std(UU[0:int(E1[1]*Coef),i])
                UU[:,i]=(T-UU[:,i])  
                UU[:,i][UU[:,i]>=0]=0
                UU[:,i][UU[:,i]<0]=1*Weights[Order[i]]
            KOKO=np.sum(UU,axis=1) 
            KOKO[KOKO<=np.sum(Weights[Order[0:len(Nets)]])/2]=0
            KOKO[KOKO>np.sum(Weights[Order[0:len(Nets)]])/2]=1        
            F1=[]
            for i in range(len(E1)-2):
                N1=KOKO[0:E1[1]]
                P1=KOKO[E1[i+1]:E1[i+2]]
                NN=len(N1)
                PP=len(P1)
                TN=NN-np.sum(N1)
                FP=np.sum(N1)
                FN=PP-np.sum(P1)
                TP=np.sum(P1)
                F1.append((2*TP)/(2*TP+FN+FP))
            F1_Mean.append(np.mean(F1))
            plt.plot(F1_Mean,'*')
            plt.ylim([min(F1_Mean)-0.02,max(F1_Mean)+0.02])
            plt.xlabel('Ensembles', fontsize=14, fontname='Times New Roman')
            plt.ylabel('Mean F1 score',fontsize=14, fontname='Times New Roman')
            plt.title('Ensemble Outcomes', fontsize=18, fontname='Times New Roman')
            plt.xticks([0, 1, 2,3,4], ['Ensemble-1','Ensemble-3','Ensemble-5','Ensemble-7','Ensemble-9'])

    '''
    This function output the SDD outcome by ensembeling all RM datasets' data classes.

    Parameters:
        Target_N (str): Target frame dataset name, ehrein it can be 'Yellow', and 'QUGS'
        
    Returns:
        nothing but a plot of ensembeling for all source domains. 
        Additionally, it outputs the AE-Zero-Shot and Blaanced supervised accuracies for comparison.

    '''
    def Ensemble_All_Comp(self,Target_N):
        W=1000
        Coef=0.25
        S=4
        folder='PT'
        cwd=self.cwd
        Rot_Data=np.load(cwd+'RM_FS_TL/A'+'_RM_S4_'+str(W)+'.npy')
        Class_N=np.cumsum(np.load(cwd+'RM_FS_TL/A'+'_RM_Classes_'+str(W)+'.npy'))
        Class_N = np.insert(Class_N, 0, 0)
        Rot_Data=Rot_Data[0:Class_N[15],:]
        Rot_Label=np.zeros((Rot_Data.shape[0],1))
        for i in range(15):
            Rot_Label[Class_N[i]:Class_N[i+1],0]=i
        #Apply 2D PCA
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(Rot_Data)
        
        Means=np.zeros((15,4))
        for i in range(15):
            Means[i,:]=np.mean(pca_data[Class_N[i]:Class_N[i+1]:100,:])
        Means[:,2]=np.sqrt((Means[:,0]-Means[0,0])**2+(Means[:,1]-Means[0,1])**2)
        dist_max=np.max(Means[:,2])
        Means[:,3]=(Means[:,2]/dist_max)
        Weights=Means[1:10,3]*1
        Order=np.argsort(Weights)
        
        
        # Target Data 
        Yc=np.load(cwd+'RM_FS_TL/'+Target_N+'_Classes_'+str(W)+'.npy')
        Yc=np.concatenate(([0],Yc))
        E1=np.cumsum(Yc)
        Yel=np.load(cwd+'RM_FS_TL/'+Target_N+'_S'+str(S)+'_'+str(W)+'.npy')
        
        [A,B]=Yel.shape
        Yel=torch.tensor(Yel,device='cuda').reshape((A,S*int(W/2))).float()

        im=0
        for RM_N in ['A','B','C']:
            F1_Mean=[]
            Ensembles=np.arange(0,len(Order),2)
            #Ensembles=[0]
            [Ytel,E1]=self.Algorith1(RM_N,Target_N)
            for ie in Ensembles:
                Nets=[]
                for t in range(ie+1):
                    RM_Class=Order[t]+1
                    Nets.append(torch.load(cwd+"RM_FS_TL/Models/"+folder+"/Model"+RM_N+str(RM_Class)+'_W_'+str(W)+".pth"))
                UU=np.zeros((len(Yel),len(Nets)))
                i=0
                for Net in Nets:
                    UU[:,i]=Net.forward(Ytel).detach().cpu().numpy().reshape((1,-1))
                    i+=1
                for i in range(len(Nets)):   
                    T=np.mean(UU[0:int(E1[1]*Coef),i])+3*np.std(UU[0:int(E1[1]*Coef),i])
                    UU[:,i]=(T-UU[:,i])  
                    UU[:,i][UU[:,i]>=0]=0
                    UU[:,i][UU[:,i]<0]=1*Weights[Order[i]]
            if im==0:
                UUP=UU*1
            else:
                UUP+=UU
            im+=1
        UUP/=3
        KOKO=np.sum(UUP,axis=1) 
        
        KOKO[KOKO<=np.sum(Weights[Order[0:len(Nets)]])/2]=0
        KOKO[KOKO>np.sum(Weights[Order[0:len(Nets)]])/2]=1        
        F1_M_E=[]
        for i in range(len(E1)-2):
            N1=KOKO[0:E1[1]]
            P1=KOKO[E1[i+1]:E1[i+2]]
            NN=len(N1)
            PP=len(P1)
            TN=NN-np.sum(N1)
            FP=np.sum(N1)
            FN=PP-np.sum(P1)
            TP=np.sum(P1)
            F1_M_E.append((2*TP)/(2*TP+FN+FP))
            
        
        #AE accuracy
        [A,B]=Yel.shape
        Yel=torch.tensor(Yel,device='cuda').reshape((A,1,int(S*W/2))).float()
        crit= nn.MSELoss(reduce=False)
        F1_AE=[]
        for T in range(len(E1)-2):
            XT=Yel[E1[0]:int(E1[1]*0.25),:,:]
            XV=Yel[int(E1[1]*0.25):int(E1[1]*0.35),:,:]
            XTE=torch.cat((Yel[int(E1[1]*0.25):int(E1[1]*1),:,:],Yel[E1[T+1]:E1[T+2],:,:]))
            F1=[]
            for ip in range(10):
                Net=torch.load(cwd+'RM_FS_TL/Models/AE/AE_'+Target_N+'_'+str(0)+'_'+str(ip+1)+'.pth')
                
                zet=0
                while(len(XT[zet*200:(zet+1)*200,:,:])>0):
                    pred=Net.forward(XT[zet*200:(zet+1)*200,:,:])
                    [A,B]=pred.shape
                    pred=pred.reshape((A,1,B))
                    loss=crit(pred,XT[zet*200:(zet+1)*200,:,:])
                    if zet==0:
                        sc=(torch.mean(torch.sum(loss,dim=2),dim=1)/int(S*W/2)).detach().cpu().numpy()
                        zet+=1
                    else:
                        zet+=1
                        sc=np.concatenate(((sc,torch.mean(torch.sum(loss,dim=2),dim=1)/int(S*W/2)).detach().cpu().numpy()))
                
                TTT=(np.mean(sc)+3*np.std(sc))
        
                zet=0
                while(len(XTE[zet*200:(zet+1)*200,:,:])>0):
                    pred=Net.forward(XTE[zet*200:(zet+1)*200,:,:])
                    [A,B]=pred.shape
                    pred=pred.reshape((A,1,B))
                    loss=crit(pred,XTE[zet*200:(zet+1)*200,:,:])
                    k=(torch.mean(torch.sum(loss,dim=2),dim=1)/int(S*W/2)).detach().cpu().numpy()
                    if zet==0:
                        sc=k
                        zet+=1
                    else:
                        zet+=1
                        sc=np.concatenate((sc,k))
            
                
                sc=TTT-sc
                N=sc[0:E1[1]-int(E1[1]*0.25)]
                P=sc[E1[1]::]
                F1_s=(2*len(P[P<0])/(2*len(P[P<0])+len(N[N<0])+len(P[P>0])))
                F1.append(F1_s)
            F1_AE.append(np.mean(F1))      
        # supervised accuracy
        F1_S=[]
        y=np.max(E1)
        y=torch.ones((y,1))
        y[0:E1[1]]=0
        y=y.to(device='cuda')
        for T in range(len(E1)-2):
            XT=torch.cat((Yel[E1[0]:int(E1[1]*0.25),:,:],Yel[E1[T+1]:E1[T+1]+int((E1[T+1]-E1[T])*0.25),:,:]))
            XV=torch.cat((Yel[int(E1[1]*0.25):int(E1[1]*0.35),:,:],Yel[E1[T+1]+int((E1[T+1]-E1[T])*0.25):E1[T+1]+int((E1[T+1]-E1[T])*0.35),:,:]))
            XTE=torch.cat((Yel[int(E1[1]*0.35):int(E1[1]*1),:,:],Yel[E1[T+1]+int((E1[T+1]-E1[T])*0.35):E1[T+1]+int((E1[T+1]-E1[T])*1),:,:]))
        
            YT=torch.cat((y[E1[0]:int(E1[1]*0.25),:],y[E1[T+1]:E1[T+1]+int((E1[T+1]-E1[T])*0.25),:]))
            YV=torch.cat((y[int(E1[1]*0.25):int(E1[1]*0.35),:],y[E1[T+1]+int((E1[T+1]-E1[T])*0.25):E1[T+1]+int((E1[T+1]-E1[T])*0.35),:]))
            YTE=torch.cat((y[int(E1[1]*0.35):int(E1[1]*1),:],y[E1[T+1]+int((E1[T+1]-E1[T])*0.35):E1[T+1]+int((E1[T+1]-E1[T])*1),:]))
            #print(E1[T+1]+int((E1[T+1]-E1[T])*0.35))
            F1_s=[]
            for ip in range(10):
                Net=torch.load(cwd+'RM_FS_TL/Models/Super/'+Target_N+'_'+str(T)+'_'+str(ip+1)+'.pth')
                predA = torch.argmax(Net.forward(XTE).softmax(dim=1),dim=1)
                #plt.plot(predA.detach().cpu().numpy())
                from sklearn.metrics import f1_score
                if f1_score(predA.detach().cpu().numpy(),YTE.detach().cpu().numpy())>0.50:
                    #print(f1_score(predA.detach().cpu().numpy(),YTE.detach().cpu().numpy()))
                    F1_s.append(f1_score(predA.detach().cpu().numpy(),YTE.detach().cpu().numpy()))
            F1_S.append(np.mean(F1_s))
        
        
        
        ### final plots
        plt.plot(F1_AE,'*', markersize=10)
        plt.plot(F1_M_E,'o', markersize=10)
        plt.plot(F1_S,'.', markersize=10)
        
        
        plt.ylim([0.4,1.01])
        plt.ylabel('F1 score', fontsize=14, fontname='Times New Roman')
        plt.xlabel('$Data Class$',fontsize=14, fontname='Times New Roman')
        plt.title('Ensemble Outcomes', fontsize=18, fontname='Times New Roman')
        lab=[]
        name=[]
        for i in range(len(E1)-2):
            name.append(i+1)
            lab.append(str(i+1))
        plt.xticks(name,lab)  
        plt.legend(['AE Zero-shot','Ensemble All','B. Supervised'])
        print('Ensemble All-Machinery mean F1 score:\t',np.mean(F1_M_E))
        print('AE mean F1 score:\t',np.mean(F1_AE))
        print('Supervised All mean F1 score:\t',np.mean(F1_S))







#Network, AE Model for the final comparison
class CustomNeuralNetworkS16(nn.Module):
    def __init__(self):
        super().__init__()
        super(CustomNeuralNetworkS16, self).__init__()
        W=1000
        S=4
        Stacked_Layers=1
        self.lstm1 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm2 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm3 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm4 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.Flatten = nn.Flatten()
        self.FC1=nn.Linear(400,100)
        self.FC2=nn.Linear(100,50)
        self.FC3=nn.Linear(50,100)
        self.FC4=nn.Linear(100,1000)
        self.FC5=nn.Linear(1000,int(S*W/2))
        self.LR1 = nn.LeakyReLU(0.2)
        self.LR2 = nn.LeakyReLU(0.2)
        self.LR3 = nn.ReLU()
        self.Sig=nn.Sigmoid()

    def forward(self, x):
        W=1000
        out1,state = self.lstm1(x[:,:,0:int(W/2)])
        out2,state = self.lstm2(x[:,:,int(W/2):2*int(W/2)])
        out3,state = self.lstm3(x[:,:,2*int(W/2):3*int(W/2)])
        out4,state = self.lstm4(x[:,:,3*int(W/2):4*int(W/2)])
        LL=torch.cat((out1,out2,out3,out4),dim=2)
        LLA=self.LR1(LL)
        outCL=self.Flatten(LLA)
        outNF=self.FC1(outCL)
        LLB=self.LR1(outNF)        
        outF=self.FC2(LLB)
        outF=self.LR3(outF)
        ####
        outF1=self.LR1(self.FC3(outF))
        outF2=self.LR1(self.FC4(outF1))
        outF3=self.LR3(self.FC5(outF2))
        return outF3
    

#Supervised Model Network for the later comparison
class CustomNeuralNetworkS15(nn.Module):
    def __init__(self):
        super().__init__()
        super(CustomNeuralNetworkS15, self).__init__()
        Stacked_Layers=1
        W=1000
        S=4
        self.lstm1 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm2 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm3 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.lstm4 = nn.LSTM(int(W/2), 100, Stacked_Layers, batch_first=True)
        self.Flatten = nn.Flatten()
        self.FC1=nn.Linear(400,100)
        self.FC2=nn.Linear(100,2)
        self.LR1 = nn.LeakyReLU(0.2)
        self.LR2 = nn.LeakyReLU(0.2)
        self.LR3 = nn.ReLU()
        self.Sig=nn.Sigmoid()

    def forward(self, x):
        W=1000
        out1,state = self.lstm1(x[:,:,0:int(W/2)])
        out2,state = self.lstm2(x[:,:,int(W/2):2*int(W/2)])
        out3,state = self.lstm3(x[:,:,2*int(W/2):3*int(W/2)])
        out4,state = self.lstm4(x[:,:,3*int(W/2):4*int(W/2)])
        LL=torch.cat((out1,out2,out3,out4),dim=2)
        LLA=self.LR1(LL)
        outCL=self.Flatten(LLA)
        outNF=self.FC1(outCL)
        LLB=self.LR1(outNF)        
        outF=self.FC2(LLB)
        outF=self.LR3(outF)
        return outF

########### 10