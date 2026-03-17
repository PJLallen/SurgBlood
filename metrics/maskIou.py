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
        union = torch.sum(self.confusionMatrix, axis=1) + torch.sum(self.confusionMatrix, axis=0) - torch.diag(self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU < float('inf')].mean()
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = torch.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.view(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass))

def load_image_as_tensor(image_path, threshold=128):
    """Load grayscale image, binarize it, and convert it to a torch tensor."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)
    return torch.from_numpy(binary_img).long()

def process_folders(folder1, folder2, num_classes, ignore_labels):
    """Process two folders of corresponding grayscale images and compute metrics."""
    metric = SegmentationMetric(num_classes)
    
    # List files in both folders
    folder1_files = sorted([f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))])
    folder2_files = sorted([f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))])
    # hist = None
    for file1, file2 in zip(folder1_files, folder2_files):
        imgPredict = load_image_as_tensor(os.path.join(folder1, file1))
        imgLabel = load_image_as_tensor(os.path.join(folder2, file2))

        # Update confusion matrix for each pair of images
        hist=metric.addBatch(imgPredict, imgLabel, ignore_labels)
    
    # Compute final metrics
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.IntersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    
    # Output results
    # print(hist)
    print('Pixel Accuracy (PA): %f' % pa)
    print('Class Pixel Accuracy (cPA):', cpa)
    print('Mean Pixel Accuracy (mPA): %f' % mpa)
    print('Intersection over Union (IoU):', IoU)
    print('Mean IoU (mIoU):', mIoU)

# Example usage
if __name__ == '__main__':
    # gt_folder = '/home/siat-hci/ZhoZhangJun/CholeBP51-imageLevel-Project/CholeBP51-image-level/test/gt'
    # pred_folder = '/home/siat-hci/ZhoZhangJun/CholeBP51-imageLevel-Project/SAM2-Adapter-mask-point/saveResult/weightSAM2-Adapter-hiera-l-point-512-unfreezeeMemory/mask'


    gt_folder = '/home/pjl307/ZZJ/CholeBleedingPoint/CholeBP51-Videos/test/videos-mask/2023-12-21patient11clip3'
    pred_folder = '/home/pjl307/ZZJ/CholeBleedingPoint/SAM2Video-ori/saveResult/weightSAM2-Adapter-video/videos-mask/2023-12-21patient11clip3'

    num_classes = 2  # Number of classes
    ignore_labels = [255]  # Labels to ignore during evaluation

    process_folders(gt_folder, pred_folder, num_classes, ignore_labels)






