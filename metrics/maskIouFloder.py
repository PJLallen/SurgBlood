import os
import torch
import cv2
import numpy as np

__all__ = ['SegmentationMetric']


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = torch.zeros((self.numClass,) * 2)  # Confusion matrix

    def pixelAccuracy(self):
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = classAcc[classAcc < float('inf')].mean()
        return meanAcc

    def IntersectionOverUnion(self):
        intersection = torch.diag(self.confusionMatrix)
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float('inf')].mean()
        return mIoU
    
    #dice metric
    def DiceCoefficient(self):
        intersection = torch.diag(self.confusionMatrix)
        total_predictions = torch.sum(self.confusionMatrix, axis=1)
        total_ground_truths = torch.sum(self.confusionMatrix, axis=0)
        
        dice = (2 * intersection) / (total_predictions + total_ground_truths)
        return dice

    def meanDiceCoefficient(self):
        dice = self.DiceCoefficient()
        mDice = dice[dice < float('inf')].mean()
        return mDice


    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):
        imgPredict = imgPredict.cpu()
        imgLabel = imgLabel.cpu()

        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        # print(imgPredict.type())
        # print(imgLabel.type())
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))


def load_image_as_tensor(image_path, threshold=128):
    """Load grayscale image, binarize it, and convert it to a torch tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #resize to 512x512
    img = cv2.resize(img, (256, 256))

    _, binary_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return torch.from_numpy(binary_img).long()


def process_folders(folder1, folder2, num_classes, ignore_labels,metric):
    # """Process two folders of corresponding grayscale images and compute metrics."""
    # metric = SegmentationMetric(num_classes)

    # List files in both folders
    folder1_files = sorted([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))])
    folder2_files = sorted([f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))])
    # hist = None
    for file1, file2 in zip(folder1_files, folder2_files):
        imgPredict = load_image_as_tensor(os.path.join(folder1, file1))
        imgLabel = load_image_as_tensor(os.path.join(folder2, file2))

        #resize to 256x256


        # Update confusion matrix for each pair of images
        hist = metric.addBatch(imgPredict, imgLabel, ignore_labels)



def traverse_directories(base_dir1, base_dir2, num_classes, ignore_labels):

    """Process folders of corresponding grayscale images and compute metrics."""
    metric = SegmentationMetric(num_classes)

    for subdir1, subdir2 in zip(os.listdir(base_dir1), os.listdir(base_dir2)):

        # print(f"Processing {subdir1} and {subdir2}")
        path1 = os.path.join(base_dir1, subdir1)
        path2 = os.path.join(base_dir2, subdir2)

        process_folders(path1, path2, num_classes, ignore_labels,metric)

        # # # # print(f"Pixel Accuracy: {metric.pixelAccuracy():.4f}")
        # # # # print(f"Class Pixel Accuracy: {metric.classPixelAccuracy()}")
        # # # # print(f"Mean Pixel Accuracy: {metric.meanPixelAccuracy():.4f}")
        # # print(f"IoU: {metric.IntersectionOverUnion()}")#IoU: tensor([0.9999, 0.9225])
        # tensor=metric.IntersectionOverUnion()
        # if 0.7<tensor[1] < 0.8:
        #     print(tensor)
        #     print(f"Processing {subdir1} and {subdir2}")
        # # # # # print(f"Mean IoU: {metric.meanIntersectionOverUnion():.4f}")
        # # # # # print(f"Dice: {metric.DiceCoefficient()}")
        # # # # # print(f"Mean Dice: {metric.meanDiceCoefficient():.4f}")
        # metric.reset()

    # print(f"Pixel Accuracy: {metric.pixelAccuracy():.4f}")
    # print(f"Class Pixel Accuracy: {metric.classPixelAccuracy()}")
    # print(f"Mean Pixel Accuracy: {metric.meanPixelAccuracy():.4f}")
    print(f"IoU: {metric.IntersectionOverUnion()}")
    # print(f"Mean IoU: {metric.meanIntersectionOverUnion():.4f}")
    print(f"Dice: {metric.DiceCoefficient()}")
    # print(f"Mean Dice: {metric.meanDiceCoefficient():.4f}")
    metric.reset()

if __name__ == '__main__':


    models = [20,18,16,14,12,10]
    
    # 循环遍历模型编号
    for model_num in models:
        print(f"Processing model-{model_num}")
        print("------------------------------------------------")

        base_gt_folder = '/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95/test/videos-mask'
        base_pred_folder = '/home/pjl307/ZZJ/CholeBleedingPoint/SAM2.1/saveResultlr120/mytest/model-{}/videos-mask'.format(model_num)
        num_classes = 2
        ignore_labels = [255]
        traverse_directories(base_gt_folder, base_pred_folder, num_classes, ignore_labels)



















    # base_gt_folder = '/mnt/sdb3/home/siat-hci/ZhoZhangJun/CholecBP113-project/CholecBP113/train/videos-mask'
    # base_pred_folder = '/mnt/sdb3/home/siat-hci/ZhoZhangJun/CholecBP113-project/SAM2-Adapter-Video-MaskandPoint-oneBranch/saveResult/sam2video-adapter-onebranch-train/videos-mask'
    # num_classes = 2
    # ignore_labels = [255]

    # traverse_directories(base_gt_folder, base_pred_folder, num_classes, ignore_labels)


