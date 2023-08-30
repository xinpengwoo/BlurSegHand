import cv2
import os
import shutil

input = "  datasets/BlurHand/blur_images/test/Capture0/ROM07_RT_Finger_Occlusions/cam400346/image23206.png"
input = input.strip()
experiments_base = "./experiments/BlurHandNet_BH/results/obj"
obj_base = "./experiments/pretrained_BlurHandNet_BH/results/obj"
video_base = "./experiments/pretrained_BlurHandNet_BH/results/video"


seg_path_parts = input.split(".")
base_name = seg_path_parts[-2].split("/")[-1]
seg_path_parts[-2] += '_seg'
seg_name = seg_path_parts[-2].split("/")[-1]
seg_path = ".".join(seg_path_parts)
seg_path_parts = seg_path.split("/")[3:]
seg_path = os.path.join(experiments_base, '/'.join(seg_path_parts))
seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(f"./results/{seg_name}.png", seg)

seg_gt_path_parts = seg_path.split(".")
seg_gt_path_parts[-2] += '_GT'
seg_gt_name = seg_gt_path_parts[-2].split("/")[-1]
seg_gt_path = ".".join(seg_gt_path_parts)
seg_gt = cv2.imread(seg_gt_path, cv2.IMREAD_GRAYSCALE)
cv2.imwrite(f"./results/{seg_gt_name}.png", seg_gt)


processed_img_path_parts = input.split(".")
processed_img_path_parts[-2] += '_processed'
processed_img_path = ".".join(processed_img_path_parts)
img = cv2.imread(processed_img_path, cv2.IMREAD_COLOR)
cv2.imwrite(f"./results/{base_name}.png", img)


seg_path_parts[-1] = base_name + '.obj'
obj_path = os.path.join(obj_base, '/'.join(seg_path_parts))
shutil.copy(obj_path, f"./results/{base_name}.obj")
seg_path_parts[-1] = base_name + '_f' + '.obj'
obj_path = os.path.join(obj_base, '/'.join(seg_path_parts))
shutil.copy(obj_path, f"./results/{base_name}_f.obj")
seg_path_parts[-1] = base_name + '_p' + '.obj'
obj_path = os.path.join(obj_base, '/'.join(seg_path_parts))
shutil.copy(obj_path, f"./results/{base_name}_p.obj")
seg_path_parts[-1] = base_name +'_video'+ '.mp4'
obj_path = os.path.join(video_base, '/'.join(seg_path_parts))
shutil.copy(obj_path, f"./results/{base_name}_video.mp4")