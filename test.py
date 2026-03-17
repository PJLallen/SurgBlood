#!/usr/bin/python3
# coding=utf-8
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sam2_video_baseline import SAM2_Video_baseline as network

from datasetVideoTest import CholeBPTest

sys.path.insert(0, '/')
sys.dont_write_bytecode = True
plt.ion()

class Test(object):
    def __init__(self, Network, data_path, weight):

        # self.test_dataset = CholeBPTest(data_path, mode='train')
        self.test_dataset = CholeBPTest(data_path, mode='test')
        self.testloader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=0)

        # Network initialization
        self.net = Network()
        self.net.train(False)
        #gpu1
        self.net = self.net.cuda()

        # Load weights
        self.net.load_state_dict(torch.load(weight))

    def save(self, save_path):
        with torch.no_grad():
            for step, sample in enumerate(self.testloader):
                images, video_name_dict, frame_names = sample['images'],sample['video_meta_dict'], sample['frame_names']

                images = images.cuda()
                images = images.view(-1, 3, 512, 512)


                shape = (512, 512)

                frame_length = images.size(0)
                # video_name
                # frame_names
                video_segments = self.net(images)

                # {'filename_or_obj': ['2023-01-17patient10']}
                video_name = video_name_dict['filename_or_obj'][0]
                print(video_name)
                # print(video_segments.keys())
                # input()


                #/videos-mask
                root_mask_path = os.path.join(save_path, 'videos-mask', video_name)#it means the path of mask,eg:saveResult/weightSAM2-Adapter-video/videos-mask/2023-01-17patient10
                root_point_path = os.path.join(save_path, 'videos-point', video_name)#it means the path of point,eg:saveResult/weightSAM2-Adapter-video/videos-point/2023-01-17patient10
                root_exited_score_path = os.path.join(save_path, 'videos-exited-score', video_name)


                if not os.path.exists(root_mask_path):
                    os.makedirs(root_mask_path)
                if not os.path.exists(root_point_path):
                    os.makedirs(root_point_path)
                if not os.path.exists(root_exited_score_path):
                    os.makedirs(root_exited_score_path)



                # print(frame_names)
                # input()


                for i in range(frame_length):
                        current_pred_mask=video_segments[i][1]['mask_logits']
                        current_pred_mask = current_pred_mask.unsqueeze(1)

                        frame_name = frame_names[i][0]
                        print(current_pred_mask.size())#torch.Size([1, 256, 256])


                        sam_pre = current_pred_mask
                        sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)  #
                        predDIS_PRE = torch.sigmoid(sam_pre[0, 0]).cpu().numpy() * 255  #
                        predDIS_PRE = np.round(predDIS_PRE).astype(np.uint8)  #

                        cv2.imwrite(os.path.join(root_mask_path, frame_name + '.jpg'), predDIS_PRE)



                        ##############################################Save point predictions#####################################
                        point_predictions=video_segments[i][1]['iou']
                        print(f"Point predictions shape: {point_predictions.shape}")  # Print shape

                        point_predictions = point_predictions.cpu().numpy()  # Move to CPU and convert to numpy
                        
                        point_x= point_predictions[0]  
                        point_y= point_predictions[1]

                        # print(f"Point predictions: {point_x}, {point_y}")  # Print the coordinates
                        # input()
                        
                
                        with open(os.path.join(root_point_path, f"{frame_name}.txt"), 'w') as f:
                            f.write(f"{point_x} {point_y}\n")  # Save the coordinates in the file




                        ##############################################Save exited score#####################################
                        exited_score_logits=video_segments[i][1]['object_score']
                        print(f"Exited score logits shape: {exited_score_logits.shape}")

                        # Apply sigmoid and convert to numpy array
                        exited_score_logits = torch.sigmoid(exited_score_logits).cpu().numpy()

                        # Save the exited score as a float with a specified decimal format
                        with open(os.path.join(root_exited_score_path, f"{frame_name}.txt"), 'w') as f:
                            f.write(f"{float(exited_score_logits[0]):.6f}")







# Demo original FPS
if __name__ == '__main__':

    models = [17]


    for model_num in models:

        weight_path = r'/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/BlooDet/save/mytest/model-{}'.format(
            model_num)


        save_path = r'/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/BlooDet/saveResult/mytestDemo/model-{}'.format(
            model_num)


        t = Test(network, data_path=r'/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95/testDemo',
                 weight=weight_path)


        t.save(save_path=save_path)