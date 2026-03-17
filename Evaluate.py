import datetime
import argparse
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.optim as optim
# import SAM2_Video_baseline as network
from sam2_video_baseline import SAM2_Video_baseline as network

from datasetVideoTrain import CholeBPTrain
from datasetVideoTest  import CholeBPTest

from metrics.maskIouFloder import SegmentationMetric


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--save_path', default="/home/pjl307/ZZJ/CholeBleedingPoint/SAM2.1/save/mytest", type=str)
    parser.add_argument('--data_path', default="/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95", type=str)
    parser.add_argument('--max_epochs', type=int, default=20, help='maximum epoch number to train')
    parser.add_argument('--base_lr', type=float, default=6e-5, help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    # parser.add_argument('--img_size', type=int, default=128, help='input patch size of network input')

    parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
    parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
    parser.add_argument('--snapshot', type=str, default=None, help='path to the snapshot')

    parser.add_argument('--test_data_path', default="/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95",type=str)

    parser.parse_args()
    return parser.parse_args()



def train(Network,testpath):
    args = parser()
    print(f"Processing {testpath}.............................................................")

    test_dataset = CholeBPTest(args.test_data_path, mode=testpath)


    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    net = Network()

    path = '/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/BlooDet/save/mytest/model-17'  # 替换为实际权重文件路径
    net.load_state_dict(torch.load(path))
    net.eval()
    net.cuda()


    #############################test############################################
    net.eval()
    ignore_labels = [255]
    metric = SegmentationMetric(2)

    frame_length_total = 0
    iou_list = []#iou
    dice_list = []#dice
    point_exist_TP = 0#count the TP point_exist score
    point_exist_total = 0#calculate the MPCK if the point is existed
    point_MPCK5_TP = 0
    point_MPCK10_TP = 0
    point_MPCK20_TP = 0

    with torch.no_grad():
        for step, sample in enumerate(testloader):
            images, masks, points, video_name_dict, frame_names = sample['images'], sample['masks'], sample['points'], sample['video_meta_dict'],sample['frame_names']
            sample['frame_names']
            images = images.cuda()
            images = images.view(-1, 3, 512, 512)
            shape = (512, 512)
            frame_length = images.size(0)
            video_segments = net(images)

            video_name = video_name_dict['filename_or_obj'][0]
            masks = masks.squeeze(0).cpu()
            points = points.squeeze(0).cpu()

            frame_length_total+=frame_length

            for i in range(frame_length):
                # mask
                current_pred_mask = video_segments[i][1]['mask_logits'].unsqueeze(1)
                frame_name = frame_names[i][0]
                # calculate iou
                sam_pre = current_pred_mask
                sam_pre = torch.sigmoid(sam_pre)

                sam_gt = masks[i].cpu()
                sam_pre = F.interpolate(sam_pre, size=shape, mode='bilinear', align_corners=False)
                sam_pre= sam_pre.squeeze(1)


                sam_pre = (sam_pre > 0.5).float()#
                sam_gt = (sam_gt > 0.5).float()

                sam_pre = (sam_pre > 0.5).int()
                sam_gt = (sam_gt > 0.5).int()
                hist = metric.addBatch(sam_pre, sam_gt, ignore_labels)

                # GT
                existed_label = ( points[i][0] != 0 and points[i][1] != 0)
                # point_exist_TP
                exited_score_logits = video_segments[i][1]['object_score']
                # Apply sigmoid and convert to numpy array
                exited_score_logits = torch.sigmoid(exited_score_logits).cpu().numpy()
                # >0.5 is 1, <0.5 is 0
                existed_pred = (exited_score_logits > 0.5).astype(int)
                # Calculate TP
                point_exist_TP += (existed_pred == existed_label.cpu().numpy()).all()

                if existed_label:
                    point_exist_total += 1

                    # point
                    pre_point = video_segments[i][1]['iou']
                    point_gt = points[i]*512
                    point_pre = pre_point*512

                    # Calculate MPCK10_TP
                    distances = torch.sqrt(torch.sum((point_gt.cpu() - point_pre.cpu()) ** 2, dim=-1))
                    if distances < (512*0.05):
                        point_MPCK5_TP += 1
                    if distances < (512*0.1):
                        point_MPCK10_TP += 1
                    if distances < (512*0.02):
                        point_MPCK20_TP += 1

    # Calculate metrics
    precision_exist = point_exist_TP / frame_length_total

    print(point_exist_total)

    MPCK5 = point_MPCK5_TP / point_exist_total
    MPCK10 = point_MPCK10_TP / point_exist_total
    MPCK20 = point_MPCK20_TP / point_exist_total
    print(f"IoU: {metric.IntersectionOverUnion()}")
    print(f"Dice: {metric.DiceCoefficient()}")

    print("precision_exist:", precision_exist)
    print("MPCK2:", MPCK20)
    print("MPCK5:", MPCK5)
    print("MPCK10:", MPCK10)


if __name__ == '__main__':
    # test_paths = [
    #     "testHemoSet",
    # ]
    test_paths = [
        "test",
    ]
    # test_paths = [
    #     "test751",
    # ]
    # test_paths = [
    #     "NIPSRebuttalDataset",
    # ]
    # test_paths = [
    #     "testGallbladder",
    #     "testCysticTriangle",
    #     "testVessel",
    #     "testGallbladderBed",
    # ]

    # 遍历测试路径并执行训练或测试函数
    for test_path in test_paths:
        train(network, testpath=test_path)  # 调用 train 函数
