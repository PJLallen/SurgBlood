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

# Define focal loss
def focal_loss(logits, mask, alpha=0.75, gamma=2.0):
    mask = F.interpolate(mask, size=logits.size()[2:], mode='bilinear')
    mask = mask.float()
    bce_loss = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    probas = torch.sigmoid(logits)
    pt = mask * probas + (1 - mask) * (1 - probas)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = mask * alpha + (1 - mask) * (1 - alpha)
    loss_focal = focal_weight * alpha_weight * bce_loss
    return loss_focal.mean()

# Define Dice Loss
def dice_loss(logits, mask, smooth=1e-6):
    mask = F.interpolate(mask, size=logits.size()[2:], mode='bilinear').float()
    logits = torch.sigmoid(logits)
    intersection = (logits * mask).sum(dim=(2, 3))
    union = logits.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

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



def train(Network):
    args = parser()

    train_dataset = CholeBPTrain(args.data_path, args.img_size,mode='train',frames_length=8,overlap=7)#fr
    test_dataset = CholeBPTest(args.test_data_path, mode='test')

    print("Total clips:", len(train_dataset))

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


    net = Network()
    net.train(True)
    net.cuda()

    # ## parameter###########################
    encoder,other, point = [], [], []
    for name, param in net.named_parameters():
        # print(name)
        if 'image_encoder' in name:
            encoder.append(param)
        elif 'sam_point_decoder'  or 'point_memory_attention' or 'edge_generator' in name:#true other all is 120x
            point.append(param)
        # elif ('sam_point_decoder' in name) or ('point_memory_attention' in name) or ('edge_generator' in name):
        #     point.append(param)
        else:
            other.append(param)

    optimizer = optim.AdamW([{'params': encoder}, {'params': point}, {'params': other}], lr=args.base_lr, betas=(0.9, 0.999), weight_decay=0.1) 

    sw = SummaryWriter(args.save_path)
    global_step = 0

    max_epoch = args.max_epochs
    for epoch in range(max_epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)) * args.base_lr
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)) * args.base_lr * 100
        optimizer.param_groups[2]['lr'] = (1 - abs((epoch + 1) / (args.max_epochs + 1) * 2 - 1)) * args.base_lr  # 5e-6

        for step, sample in enumerate(trainloader):
            image, mask, edge, point, video_meta_dict = sample['image'], sample['mask'], sample['edge'], sample[
                'point'], sample['video_meta_dict']

            images, masks, edges = image.float().cuda(), mask.float().cuda(), edge.float().cuda()

            points = point.float().cuda()

            images = images.view(-1, 3, args.img_size, args.img_size)  # 384
            masks = masks.view(-1, 1, args.img_size, args.img_size)
            edges = edges.view(-1, 1, args.img_size, args.img_size)
            points = points.view(-1, 2)

            frame_length = images.size(0)

            # Forward pass
            # mask_pred = net(image, multimask_output=False, image_size=args.img_size)
            video_segments = net(images)

            pre_masks = []
            pre_edges = []
            pre_point = []  # iou now is point in the code
            pre_existed_score_logits = []

            # Assuming 'video_segments' is indexed first by frame index 'i' and then object index '1'
            for i in range(frame_length):  # Assuming you have 8 segments as your previous code implied
                if 1 in video_segments[i]:
                    pre_masks.append(video_segments[i][1]['mask_logits'])
                    pre_point.append(video_segments[i][1]['iou'])
                    pre_existed_score_logits.append(video_segments[i][1]['object_score'])

                    pre_edges.append(video_segments[i][1]['edge_logits'])

                else:
                    print(f"Key 1 not found in video_segments[{i}]")

            # Convert lists to PyTorch tensors by stacking
            mask_pred = torch.stack(pre_masks, dim=0)
            pre_point = torch.stack(pre_point, dim=0)
            existed_score_logits = torch.stack(pre_existed_score_logits, dim=0)

            edge_pred = torch.stack(pre_edges, dim=0)

            mask = mask.squeeze(0).cuda()
            edge = edge.squeeze(0).cuda()

            # Focal loss + Dice loss
            loss_focal = focal_loss(mask_pred, mask)
            loss_dice = dice_loss(mask_pred, mask)
            lossmask = loss_focal + loss_dice

            # Focal loss + Dice loss
            loss_focal_edge = focal_loss(edge_pred, edge)
            loss_dice_edge = dice_loss(edge_pred, edge)
            lossmask_edge = loss_focal_edge + loss_dice_edge

            # Check for existence of points in each sample
            existed_label = (points != 0).any(dim=1).float().cuda()
            # Calculate binary cross-entropy loss
            lossExsited = F.binary_cross_entropy_with_logits(existed_score_logits, existed_label.unsqueeze(1))

            # Calculate point loss based on whether each sample in the batch has all zeros
            lossPoint = torch.tensor(0.0, device=points.device)  # Initialize lossPoint to zero tensor

            for i in range(points.size(0)):  # Iterate over each sample in the batch
                if not torch.all(points[i] == 0):  # Check if the sample is not all zeros
                    lossPoint += F.smooth_l1_loss(pre_point[i].unsqueeze(0), points[i].unsqueeze(0))

            # ggpu1
            loss = lossmask + lossExsited + lossPoint + 0.5 * lossmask_edge

            if lossmask > 10 and epoch > 17:
                print("lossmask:", lossmask)
                print("video_meta_dict:", video_meta_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment global step
            global_step += 1

            # Log learning rate and losses
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('losses', {
                'total_loss': loss.item(),
                'mask_loss': lossmask.item(),
                'lossPoint': lossPoint.item(),
                'existed_loss': lossExsited.item()
            }, global_step=global_step)

            # Print detailed status every 75 steps
            if step % 200 == 0:
                print(
                    f'{datetime.datetime.now()} | Step: {global_step}/{epoch + 1}/{max_epoch} | '
                    f'Total Loss={loss.item():.6f}, Mask Loss={lossmask.item():.6f}, '
                    f'Point Loss={lossPoint.item():.6f}, Existed Loss={lossExsited.item():.9f}, '
                )

        torch.save(net.state_dict(), args.save_path + f'/model-{epoch + 1}')
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
        MPCK5 = point_MPCK5_TP / point_exist_total
        MPCK10 = point_MPCK10_TP / point_exist_total
        MPCK20 = point_MPCK20_TP / point_exist_total
        print("evaluate_epoch---------------:", epoch+1)
        print(f"IoU: {metric.IntersectionOverUnion()}")
        print(f"Dice: {metric.DiceCoefficient()}")

        print("precision_exist:", precision_exist)
        print("MPCK2:", MPCK20)
        print("MPCK5:", MPCK5)
        print("MPCK10:", MPCK10)


if __name__ == '__main__':
    train(network)
