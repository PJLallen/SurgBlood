import torch
import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import time
import numpy as np
import PIL.Image
from pwcnet.run import Network

class OpticalFlowEstimator(torch.nn.Module):
    def __init__(self, model_path='network-default.pytorch'):
        super(OpticalFlowEstimator, self).__init__()

        self.netNetwork = Network().cuda().eval()
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        """Load pre-trained model weights."""
        self.netNetwork.load_state_dict({strKey.replace('module', 'net'): tenWeight
                                         for strKey, tenWeight in torch.load(self.model_path).items()})

    def forward(self, tenFirst, tenSecond):
        """
        Given two input tensors `tenFirst` and `tenSecond`, estimate the optical flow.
        Args:
            tenFirst: The first input image tensor (shape: [3, H, W]).
            tenSecond: The second input image tensor (shape: [3, H, W]).
        Returns:
            tenFlow: The estimated optical flow (shape: [2, H, W]).
        """

        # Preprocess inputs (resize and normalize)
        assert (tenFirst.shape[1] == tenSecond.shape[1])
        assert (tenFirst.shape[2] == tenSecond.shape[2])

        intWidth = tenFirst.shape[2]
        intHeight = tenFirst.shape[1]

        assert(intWidth == 512) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
        assert(intHeight == 512) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue


        tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
        tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
                                                               size=(intPreprocessedHeight, intPreprocessedWidth),
                                                               mode='bilinear', align_corners=False)
        tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
                                                                size=(intPreprocessedHeight, intPreprocessedWidth),
                                                                mode='bilinear', align_corners=False)

        # print(tenFirst.shape)
        # print(tenSecond.shape)

        # Estimate optical flow using the network

        tenFlow = 20.0 * torch.nn.functional.interpolate(input=self.netNetwork(tenPreprocessedFirst, tenPreprocessedSecond),
                                                         size=(intHeight, intWidth), mode='bilinear',
                                                         align_corners=False)

        # print(tenFlow.shape)
        # input(222)

        # Rescale the flow to match the original input dimensions
        tenFlow[:, 0, :, :] *= float(tenFirst.shape[2]) / float(tenFlow.shape[3])
        tenFlow[:, 1, :, :] *= float(tenFirst.shape[1]) / float(tenFlow.shape[2])

        return tenFlow[0, :, :, :].cpu()



    def estimate(self, tenFirst, tenSecond):
        """
        Estimate the optical flow given two input tensors.
        Args:
            tenFirst: The first input image tensor.
            tenSecond: The second input image tensor.
        Returns:
            The estimated optical flow tensor.
        """
        return self.forward(tenFirst, tenSecond)


# Example usage:
if __name__ == '__main__':
    # Load your images as torch tensors
    tenFirst = torch.FloatTensor(np.ascontiguousarray(
        np.array(PIL.Image.open('./images/first.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    tenSecond = torch.FloatTensor(np.ascontiguousarray(
        np.array(PIL.Image.open('./images/second.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    # Instantiate the optical flow estimator
    estimator = OpticalFlowEstimator(model_path='network-default.pytorch')

    print(tenFirst.shape)
    print(tenSecond.shape)

    # Estimate optical flow
    tenOutput = estimator.estimate(tenFirst, tenSecond)
    print(tenOutput.shape)

    # Save the output flow to a file
    with open('./out.flo', 'wb') as objOutput:
        np.array([80, 73, 69, 72], np.uint8).tofile(objOutput)
        np.array([tenOutput.shape[2], tenOutput.shape[1]], np.int32).tofile(objOutput)
        np.array(tenOutput.numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)
