import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms

def main():
    # seg_path = "/home/zzq/Xinpeng/BlurHand_RELEASE/datasets/BlurHand/blur_images/train/Capture7/0011_aokay/cam400064/image4471-var-mask.png"
    # seg2_path = "/home/zzq/Xinpeng/BlurHand_RELEASE/datasets/BlurHand/blur_images/train/Capture7/0011_aokay/cam400064/image4489-var-mask.png"
    # seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    # seg2 = cv2.imread(seg2_path, cv2.IMREAD_GRAYSCALE)

        # Create a sample binary image tensor as a 2D PyTorch tensor
    binary_image_tensor = torch.tensor([[0, 1, 0],
                                        [1, 0, 1],
                                        [0, 1, 0]], dtype=torch.uint8)

    # Convert the tensor to a NumPy array
    binary_image_np = binary_image_tensor.numpy()
    binary_image_np[binary_image_np>0] = 255
    binary_image_np[binary_image_np<=0] = 0
    # Save the binary image using OpenCV
    output_path = './binary_image.png'  # Change the extension based on your desired format
    cv2.imwrite(output_path, binary_image_np)
    # transform = transforms.ToTensor()
    # eps = 1e-7
    # batch_size= 24
    # inputs = torch.zeros(batch_size,seg.shape[0],seg.shape[1]) / 255
    # targets = transform(np.repeat(seg[:, :, np.newaxis], batch_size, axis=2))
    # targets[12:,:,:] = transform(np.repeat(seg2[:, :, np.newaxis], batch_size / 2, axis=2))
    # inputs[targets==1] = 1
    # # inputs = inputs.view(batch_size, -1)
    # # targets = targets.view(batch_size, -1)
    # # intersection = (inputs * targets).sum()
    
    # # dice_loss = 1 - (2.*intersection + eps) / (inputs.sum() + targets.sum() + eps)

    # bce_loss = nn.BCELoss(reduction='none')
    # loss = bce_loss(inputs, targets)
    # # 打印张量的值域
    # min_value = torch.min(loss)
    # max_value = torch.max(loss)

    # print("最小值:", min_value)
    # print("最大值:", max_value)
    # print(dice_loss)

if __name__ == "__main__":
    main()