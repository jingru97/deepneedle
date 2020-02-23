#System
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import time
from scipy.spatial.distance import directed_hausdorff
from numpy.core.umath_tests import inner1d
from skimage.io import imread, imsave
#Torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as standard_transforms
from torchvision.models import resnet18
#Customs
#from ptsemseg.models import get_model

#from linknet_old import LinkNet as LinkNet
from linknet_spp2 import LinkNet as LinkNet
from binary import hd95

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ckpt_path = './ckpt_linknet_4nov_spp14e'
#ckpt_path = './ckpt_linknet_4nov_nospp_batch6'
#ckpt_linknet_verify 
#ckpt_linknet_batch6 
#ckpt_linknet_datasetnew
#ckpt_linknet_4nov_spp14e
#ckpt_linknet_4nov_nospp_batch20
#ckpt_linknet_4nov_nospp_batch6

args = {
    'exp_name': 'TSHR-LinkNet',
    'snapshot': '',
    'num_class': 2,
    'batch_size':1, #original:8
    'num_gpus':1,
}
def HausdorffDist(A,B):
    D_mat = np.sqrt(inner1d(A, A)[np.newaxis].T + inner1d(B, B) - 2 * (np.dot(A, B.T)))
    # Find DH
    dH = np.max(np.array([np.max(np.min(D_mat, axis=0)), np.max(np.min(D_mat, axis=1))]))
    return (dH)

def dice(pred, label):
    dice_val = np.float(np.sum(pred[label == 1] == 1)) * 2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_val

def specificity(TP, TN, FP, FN):
    return TN / (FP + TN)


def sensitivity(TP, TN, FP, FN):
    return TP / (TP + FN)

class TSHRDataset_test(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.img_anno_pairs = glob(img_dir)
        


    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _target = Image.open(self.img_anno_pairs[index]).convert('L')
        _img = Image.open(self.img_anno_pairs[index][:-9] + '.png').convert('RGB')
        #_img = torch.from_numpy(np.array(_img).transpose(2,0,1)).float()
        _target = np.array(_target)
        _target[_target == 255] = 1
        _target = torch.from_numpy(np.array(_target)).long()
        #_img = _img.resize((512, 256), Image.BILINEAR)
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        return _img, _target, self.img_anno_pairs[index]

if __name__ == '__main__':
    #img_dir = '/media/mobarak/data/Datasets/TUMOR_SUR_HR/Test/img_seg/**[0-9000].jpg'
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5/**_mask.png'
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_raw/**_mask.png'
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_original/**_mask.png'
    #img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_reduced/**_mask.png'
    img_dir = '/home/mmlab/jiayi/data21_dataset_v5_patient1to8_combined_splitted/test_v5_reduced2/**_mask.png'
    
    dataset = TSHRDataset_test(img_dir=img_dir)
    print("len",len(dataset), 'using', ckpt_path)
    
    test_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False, num_workers=2)
    #model = get_model("pspnet", n_classes=2)
    model = LinkNet(n_classes=2)
    gpu_ids = range(args['num_gpus'])
    model = torch.nn.parallel.DataParallel(model, device_ids=gpu_ids)
    model = model.cuda()
    Best_Dice = 0
    Best_epoch=0
    for epochs in range(171,172): #nospp_batch6:187, spp14e:171
        args['snapshot'] = 'epoch_' + str(epochs) + '.pth.tar'
#        model.load_state_dict(torch.load(os.path.join(ckpt_path, args['exp_name'], args['snapshot'])))
        model_path=os.path.join(ckpt_path, args['exp_name'], args['snapshot'])
        model.load_state_dict(torch.load(model_path))
        #model.load_state_dict(torch.load('/home/mmlab/jiayi/pytorch/ckpt/TSHR-LinkNet/epoch_93_best.pth.tar'))
        
        model.eval()
        mdice = []
        mspecificity = []
        msensitivity = []
        mhausdorff = []
        haus = []
        mytime = []
        mymin = 10000
        dice30=[]
        dice40=[]
        dice50=[]
        dice60=[]
        dice70=[]
        dice75=[]
        dice80=[]
        dice90=[]
        dice100=[]
        
        for batch_idx, data in enumerate(test_loader):
            inputs, labels, mpath = data
            inputs = Variable(inputs).cuda()
            #print(inputs.size())
            t0=time.time()
            outputs = model(inputs)
            t1=time.time()
            mytime.append(t1-t0)
            #print(t1-t0)
            #print('current:', t1-t0, 'min:',mymin)
            img_pred = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
            labels = np.array(labels)
            for dice_idx in range(0,img_pred.shape[0]):
                if(np.max(labels[dice_idx])==0):
                    continue
                dice_value=dice(img_pred[dice_idx], labels[dice_idx])
                

                mdice.append(dice_value)
                print(mpath,': ',dice_value)

                #other metrics
                A = np.logical_and(img_pred[dice_idx], labels[dice_idx])
                tp = float(A[A > 0].shape[0])
                tn = float(A[A == 0].shape[0])
                B = img_pred[dice_idx] - labels[dice_idx]
                fp = float(B[B > 0].shape[0])
                fn = float(B[B < 0].shape[0])
                mspecificity.append(specificity(tp, tn, fp, fn))
                msensitivity.append(sensitivity(tp, tn, fp, fn))
                #mhausdorff.append(HausdorffDist(img_pred[dice_idx], labels[dice_idx]))
                #mhausdorff.append(directed_hausdorff(img_pred[dice_idx], labels[dice_idx])[0])
                mhausdorff.append(hd95(img_pred[dice_idx], labels[dice_idx]))

                img_pred[dice_idx][img_pred[dice_idx] == 1] = 255
                pred_dir='preds_'+ckpt_path[7:]+'_reducedTest/' #'linknet_spp8/'
                
                seg_path = pred_dir+os.path.basename(mpath[dice_idx])
                
                if not os.path.exists(pred_dir):
                    os.mkdir(pred_dir)
                for x in [30, 40, 50, 60, 70, 75, 80, 90, 100]:
                    pred_dir_sub=pred_dir+str(x)
                    if not os.path.exists(pred_dir_sub):
                        pass
                        os.mkdir(pred_dir_sub)
                        print('created',pred_dir_sub)
                #print('seg',seg_path)
                #imsave(seg_path, img_pred[dice_idx])
        '''
                labels[dice_idx]=labels[dice_idx]*255
                pred_path=pred_dir+os.path.basename(mpath[dice_idx])
                imsave(pred_path,img_pred[dice_idx])
                label_path=pred_dir+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png')
                imsave(label_path,labels[dice_idx])
                print('seg1',pred_path, label_path)
                if dice_value<0.3:
                    dice30.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'30/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'30/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.4 and dice_value>=0.3:
                    dice40.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'40/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'40/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.5 and dice_value>=0.4:
                    dice50.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'50/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'50/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.6 and dice_value>=0.5:
                    dice60.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'60/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'60/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.7 and dice_value>=0.6:
                    dice70.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'70/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'70/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.75 and dice_value>=0.7:
                    dice75.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'75/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'75/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.8 and dice_value>=0.75:
                    dice80.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'80/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'80/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value<0.9 and dice_value>=0.8:
                    dice90.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'90/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'90/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                elif dice_value>=0.9:
                    dice100.append(str(dice_idx)+' '+mpath[dice_idx]+' '+str(dice_value)+'\r\n')
                    imsave(pred_dir+'100/'+os.path.basename(mpath[dice_idx]),img_pred[dice_idx])
                    imsave(pred_dir+'100/'+os.path.basename(mpath[dice_idx]).replace('_mask.png','.png'),labels[dice_idx])
                
                

        
        with open("dice_linknet_"+ckpt_path[7:]+".txt","w") as f:
            f.writelines("<0.3\r\n")
            f.writelines(dice30)
            f.writelines("<0.4\r\n")
            f.writelines(dice40)
            f.writelines("<0.5\r\n")
            f.writelines(dice50)
            f.writelines("<0.6\r\n")
            f.writelines(dice60)
            f.writelines("<0.7\r\n")
            f.writelines(dice70)
            f.writelines("<0.75\r\n")
            f.writelines(dice75)
            f.writelines("<0.8\r\n")
            f.writelines(dice80)
            f.writelines("<0.9\r\n")
            f.writelines(dice90)
            f.writelines("<=1.0\r\n")
            f.writelines(dice100)
        



        
        
        if np.mean(mdice)>Best_Dice:
            Best_Dice = np.mean(mdice)
            Best_epoch = epochs;
        #print("mdice",mdice,"len",len(mdice),len(dice30),len(dice40),len(dice50),len(dice60),len(dice70),len(dice100))
        #print(str(epochs) + ':' +'No of test:'+ str(len(mdice)) +" Dice:"+  str(np.mean(mdice)) +'   Best='+str(Best_epoch)+':'+str(Best_Dice))
        print('{:2d}: No of test:{}  Dice:{:.4f} Best={} : {:.4f}'.format(epochs, len(mdice), np.mean(mdice), Best_epoch, Best_Dice))
        print('Dice:',np.mean(mdice), 'Hausdorff:',np.mean(mhausdorff), 'Sensitivity:',np.mean(msensitivity),'Specificity:',np.mean(mspecificity), 'Avg Time(ms):',np.mean(mytime)*1000, 'fps:',(1.0/np.mean(mytime)))
        print("dice_linknet_"+ckpt_path[7:]+".txt")
        '''
