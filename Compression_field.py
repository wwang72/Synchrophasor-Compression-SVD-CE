import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
def compress(currentFilename,signal):
    tolerance_rate = 0.05
    fileName  = currentFilename
    data_array = np.array(pd.read_csv(fileName))
    evaluate_data = data_array-60
    n_data_diff = normalize(evaluate_data,axis=0,norm='l1')
    n_data_diff_mean = np.mean(n_data_diff,axis=1)
    
    diff_to_nominal = abs((n_data_diff.T-n_data_diff_mean).T)
    entropy = -diff_to_nominal*np.log(1-diff_to_nominal)
    weighted_entropy_list = np.max(entropy,axis=1)
    window_size = 5
    
    counter = 0
    start = 0
    end = 0
    highEntropyPeriods = []
    tolerance = 10
    totalSamples = weighted_entropy_list.shape[0]
    #print(weighted_entropy_list)
    max_entropy = entropy.max()
    print(max_entropy)
    for i in range(1,totalSamples):
        #print(counter)
        if(weighted_entropy_list[i]>max_entropy/10):
            if(counter==0):
                start = i
            counter = tolerance
        else:
            if(counter>0):
                counter = counter - 1
                if(counter==0):
                    end = i
                    rangeList = []
                    rangeList.append(start)
                    rangeList.append(end)
                    highEntropyPeriods.append(rangeList)
    #print(highEntropyPeriods)
    if(len(highEntropyPeriods)==0):
        maximum_error_threshold = max(abs(np.max(n_data_diff)*tolerance_rate),abs(np.min(n_data_diff)*tolerance_rate))
        print(maximum_error_threshold)
        u,s,vh = np.linalg.svd(n_data_diff, full_matrices = False)
        for reducedDimension in range(1,n_data_diff.shape[1]+1):
            new_u = u[:,0:int(reducedDimension)]
            new_s = s[0:int(reducedDimension)]
            new_vh = vh[0:int(reducedDimension),:]
            np.savetxt('middle_result'+str(i)+'u'+fileName+'.csv',new_u,delimiter = ',')
            np.savetxt('middle_result'+str(i)+'s'+fileName+'.csv',new_s,delimiter = ',')
            np.savetxt('middle_result'+str(i)+'vh'+fileName+'.csv',new_vh,delimiter = ',')
            recon = np.dot(new_u*new_s,new_vh)
            if(abs(recon-n_data_diff).max()<=maximum_error_threshold):
                break;
        print('CR:'+str(n_data_diff.shape[1]/reducedDimension))
        np.savetxt('recon_mya_'+signal+currentFilename,recon,delimiter = ',')
        np.savetxt('original_'+signal+currentFilename,n_data_diff,delimiter = ',')
        return;

    merge_window=10
    highEntropyPeriods = np.array(highEntropyPeriods)
    if(highEntropyPeriods.shape[0]==0):
        mergedPeriods = np.array([[0,n_data_diff.shape[0]-1]])
    else:
        mergedPeriods = np.array([highEntropyPeriods[0]])
    current_merge_index = 0
    for i in range(1,highEntropyPeriods.shape[0]):
        if(highEntropyPeriods[i][0]<mergedPeriods[current_merge_index][1]+merge_window):
            mergedPeriods[current_merge_index][1]=highEntropyPeriods[i][1]
        else:
            mergedPeriods = np.append(mergedPeriods,[highEntropyPeriods[i]],axis = 0)
            current_merge_index = current_merge_index + 1
    
    for i in range(0,mergedPeriods.shape[0]):
        if(i==0 and mergedPeriods[i][0]>0):
            allPeriods = np.array([[0,mergedPeriods[i][0]-1, np.median(entropy[0:mergedPeriods[i][0]-1]),entropy[0:mergedPeriods[i][0]-1].mean(),entropy[0:mergedPeriods[i][0]-1].max()]])
        if(mergedPeriods[i][0]-allPeriods[allPeriods.shape[0]-1][1]>1):
            allPeriods = np.append(allPeriods, [[allPeriods[allPeriods.shape[0]-1][1]+1,mergedPeriods[i][0]-1,np.median(entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)]),entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)].mean(),entropy[int(allPeriods[allPeriods.shape[0]-1][1]+1):int(mergedPeriods[i][0]-1)].max()]], axis = 0)
        allPeriods = np.append(allPeriods, [[mergedPeriods[i][0],mergedPeriods[i][1],np.median(entropy[mergedPeriods[i][0]:mergedPeriods[i][1]]),entropy[mergedPeriods[i][0]:mergedPeriods[i][1]].mean(),entropy[mergedPeriods[i][0]:mergedPeriods[i][1]].max()]], axis = 0)
        if (i==mergedPeriods.shape[0]-1 and mergedPeriods[i][1]+1<entropy.shape[0]-1):
            allPeriods = np.append(allPeriods,[[mergedPeriods[i][1]+1,entropy.shape[0]-1,np.median(entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)]),entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)].mean(),entropy[int(mergedPeriods[i][1]+1):int(entropy.shape[0]-1)].max()]],axis = 0)
    #print(allPeriods)
    mean_diffs = []
    max_diffs = []
    min_diffs = []
    tolerance_rate = 0.05
    total_points = 0
    for i in range(0,allPeriods.shape[0]):
        this_chunk = n_data_diff[int(allPeriods[i][0]):int(allPeriods[i][1])+1]
        maximum_error_threshold = max(abs(np.max(n_data_diff)*tolerance_rate),abs(np.min(n_data_diff)*tolerance_rate))
        u,s,vh = np.linalg.svd(this_chunk, full_matrices = False)
        for reducedDimension in range(0,n_data_diff.shape[1]+1):
            new_u = u[:,0:int(reducedDimension)]
            new_s = s[0:int(reducedDimension)]
            new_vh = vh[0:int(reducedDimension),:]
            np.savetxt('middle_result'+str(i)+'u'+fileName+'.csv',new_u,delimiter = ',')
            np.savetxt('middle_result'+str(i)+'s'+fileName+'.csv',new_s,delimiter = ',')
            np.savetxt('middle_result'+str(i)+'vh'+fileName+'.csv',new_vh,delimiter = ',')
            recon = np.dot(new_u*new_s,new_vh)
            if(abs(recon-this_chunk).max()<maximum_error_threshold):
                break;
        #print(reducedDimension)
        total_points = total_points+this_chunk.shape[0]*n_data_diff.shape[1]/reducedDimension
        if(i==0):
            entropy_to_dimension=np.array([np.append(allPeriods[i][2:],reducedDimension)])
        else:
            entropy_to_dimension = np.append(entropy_to_dimension,[np.append(allPeriods[i][2:],reducedDimension)],axis=0)
         #start from 2 because 0 and 1 are periods start and end
        new_u = u[:,0:int(reducedDimension)]
        new_s = s[0:int(reducedDimension)]
        new_vh = vh[0:int(reducedDimension),:]
        recon = np.dot(new_u*new_s,new_vh)
        print("Reconstructed "+str(recon.shape[0])+" samples."+'at rate '+str(reducedDimension))
        if(i==0):
            recon_full = recon
        else:
            recon_full = np.append(recon_full,recon,axis = 0)
        diff = abs(recon-this_chunk)
        [].append(diff.mean())
        max_diffs.append(diff.max())
        mean_diffs.append(diff.mean())
        min_diffs.append(diff.min())
        #np.save(fileName+'_u'+str(i),new_u)
        #np.save(fileName+'_s'+str(i),new_s)
        #np.save(fileName+'_vh'+str(i),new_vh)
    print('max_diff:'+str(np.max(max_diffs)))
    print('mean_diff'+str(np.mean(mean_diffs)))
    print('CR:'+str(total_points/(n_data_diff[0].shape[0]-allPeriods[-1][0])))
    np.savetxt('recon_mya_'+signal+currentFilename,recon_full,delimiter = ',')
    np.savetxt('original_'+signal+currentFilename,n_data_diff,delimiter = ',')


def compressPaper(currentFilename,signal):
    tolerance_rate = 0.05
    fileName  = currentFilename
    data_array = np.array(pd.read_csv(fileName))
    evaluate_data = data_array-60
    n_data_diff = normalize(evaluate_data,axis=0,norm='l1')
    n_data_diff_mean = np.mean(n_data_diff,axis=1)
    moving_window = 10
    limit = -1
    scd_s = -1
    scd_e = -1
    count = 0
    for i in range(moving_window,n_data_diff_mean.shape[0]):
        if(limit==-1):
            limit = abs(n_data_diff_mean[i]-np.mean(n_data_diff_mean[i-moving_window:i]))
        elif(abs(n_data_diff_mean[i]-np.mean(n_data_diff_mean[i-moving_window:i]))>5*limit):
            scd_s = i
        elif(scd_s != -1 and abs(n_data_diff_mean[i]-np.mean(n_data_diff_mean[i-moving_window:i]))<2*limit):
            if(count<10):
                count = count+1
            else:
                scd_e = i
                break;
    highEntropyPeriods = []
    if(scd_s==-1):
        highEntropyPeriods.append([0,n_data_diff_mean.shape[0]])
    else:
        if(scd_e==-1):
            highEntropyPeriods.append([scd_s,n_data_diff_mean.shape[0]])
        else:
            highEntropyPeriods.append([scd_s,scd_e])

    #diff_to_nominal = abs((n_data_diff.T-n_data_diff_mean).T)
    #entropy = -diff_to_nominal*np.log(1-diff_to_nominal)
    #weighted_entropy_list = np.max(entropy,axis=1)
    #window_size = 5
    
    #counter = 0
    #start = 0
    #end = 0
    #highEntropyPeriods = []
    #tolerance = 10
    #totalSamples = weighted_entropy_list.shape[0]
    #print(weighted_entropy_list)
    #max_entropy = entropy.max()
    #print(max_entropy)
    #for i in range(1,totalSamples):
    #    #print(counter)
    #    if(weighted_entropy_list[i]>max_entropy/10):
     #       if(counter==0):
      #          start = i
       #     counter = tolerance
        #else:
         #   if(counter>0):
          #      counter = counter - 1
           #     if(counter==0):
            #        end = i
             #       rangeList = []
              #      rangeList.append(start)
               #     rangeList.append(end)
                #    highEntropyPeriods.append(rangeList)
    #print(highEntropyPeriods)
    merge_window=10
    highEntropyPeriods = np.array(highEntropyPeriods)
    if(highEntropyPeriods.shape[0]==0):
        mergedPeriods = np.array([[0,n_data_diff.shape[0]-1]])
    else:
        mergedPeriods = np.array([highEntropyPeriods[0]])
    current_merge_index = 0
    for i in range(1,highEntropyPeriods.shape[0]):
        if(highEntropyPeriods[i][0]<mergedPeriods[current_merge_index][1]+merge_window):
            mergedPeriods[current_merge_index][1]=highEntropyPeriods[i][1]
        else:
            mergedPeriods = np.append(mergedPeriods,[highEntropyPeriods[i]],axis = 0)
            current_merge_index = current_merge_index + 1
    
    for i in range(0,mergedPeriods.shape[0]):
        if(i==0 and mergedPeriods[i][0]>0):
            allPeriods = np.array([[0,mergedPeriods[i][0]-1, 0,0,0,]])
        if(i==0 and mergedPeriods[i][0]==0):
            allPeriods = np.array([[0,mergedPeriods[i][1]-1, 0,0,0,]])
        if(mergedPeriods[i][0]-allPeriods[allPeriods.shape[0]-1][1]>1):
            allPeriods = np.append(allPeriods, [[allPeriods[allPeriods.shape[0]-1][1]+1,mergedPeriods[i][0]-1,0,0,0,]], axis = 0)
        allPeriods = np.append(allPeriods, [[mergedPeriods[i][0],mergedPeriods[i][1], 0,0,0,]], axis = 0)
        if (i==mergedPeriods.shape[0]-1 and mergedPeriods[i][1]+1<n_data_diff_mean.shape[0]-1):
            allPeriods = np.append(allPeriods,[[mergedPeriods[i][1]+1,n_data_diff_mean.shape[0]-1, 0,0,0,]],axis = 0)
    #print(allPeriods)
    mean_diffs = []
    max_diffs = []
    min_diffs = []
    total_points = 0
    print('compressing via paper')
    print(allPeriods)
    if(allPeriods.shape[0]==0):
        allPeriods = np.array([[0,n_data_diff.shape[0],0,0,0]])
    for i in range(0,allPeriods.shape[0]):
        this_chunk = n_data_diff[int(allPeriods[i][0]):int(allPeriods[i][1])+1]
        u,s,vh = np.linalg.svd(this_chunk, full_matrices = False)
        if(allPeriods[i][1]-allPeriods[i][0]<60):
            score_threshold = 0.95
        else:
            score_threshold = 0.8
        reducedDimension = 1
        for n_scores in range(0,n_data_diff.shape[1]+1):
            #print(sum(s[0:n_scores])/sum(s))
            if(sum(s[0:n_scores])/sum(s)>score_threshold):
                reducedDimension = n_scores
                break;
            new_u = u[:,0:int(reducedDimension)]
            new_s = s[0:int(reducedDimension)]
            new_vh = vh[0:int(reducedDimension),:]
            recon = np.dot(new_u*new_s,new_vh)
        total_points = total_points+this_chunk.shape[0]*n_data_diff.shape[1]/reducedDimension
        if(i==0):
            entropy_to_dimension=np.array([np.append(allPeriods[i][2:],reducedDimension)])
        else:
            entropy_to_dimension = np.append(entropy_to_dimension,[np.append(allPeriods[i][2:],reducedDimension)],axis=0)
         #start from 2 because 0 and 1 are periods start and end
        new_u = u[:,0:int(reducedDimension)]
        new_s = s[0:int(reducedDimension)]
        new_vh = vh[0:int(reducedDimension),:]
        recon = np.dot(new_u*new_s,new_vh)
        print("Reconstructed "+str(recon.shape[0])+" samples."+'at rate '+str(reducedDimension))
        if(i==0):
            recon_full = recon
        else:
            recon_full = np.append(recon_full,recon,axis = 0)
        diff = abs(recon-this_chunk)
        [].append(diff.mean())
        max_diffs.append(diff.max())
        mean_diffs.append(diff.mean())
        min_diffs.append(diff.min())
        #np.save(fileName+'_u'+str(i),new_u)
        #np.save(fileName+'_s'+str(i),new_s)
        #np.save(fileName+'_vh'+str(i),new_vh)
    print('max_diff:'+str(np.max(max_diffs)))
    print('mean_diff'+str(np.mean(mean_diffs)))
    print('CR:'+str(n_data_diff.shape[1]/reducedDimension))
    np.savetxt('recon_paper_'+signal+currentFilename,recon_full,delimiter=',')
    np.savetxt('original_'+signal+currentFilename,n_data_diff,delimiter=',')
    if(entropy_to_dimension.shape[0]>0):
        entropy_to_dimension = entropy_to_dimension[1:] # exclude the first chunk because it will contains both event and ambient data
    
    np.save('entropy_to_dimension_training'+fileName,entropy_to_dimension)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,1)
    axs.plot(n_data_diff)
    plt.show()