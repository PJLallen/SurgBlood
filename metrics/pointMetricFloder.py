import numpy as np
import os

def read_point_file(file_path):
    # print(file_path)
    with open(file_path, 'r') as file:
        point = np.array([float(num) for num in file.readline().strip().split()])
    return point

def read_existed_score_file(file_path):
    with open(file_path, 'r') as file:
        score = float(file.readline().strip())
    return score

def compute_metrics(
    gt_point: np.ndarray,
    gt_existeded: np.ndarray,
    pre_point: np.ndarray,
    pre_existeded: float,
) -> dict:
    gt_point = gt_point * 256
    pre_point = pre_point * 256

    # pixel_thresholds = [1, 2, 4, 8,16]
    # pixel_thresholds = [10]
    # pixel_thresholds = [256*0.03]
    pixel_thresholds = [256*0.2]
    # pixel_thresholds = [40]
    # pixel_thresholds = [80]
    # pixel_thresholds = [10, 20, 40, 80]

    pre_existeded_binary = np.array(pre_existeded > 0.5, dtype=np.int32)

    metrics = {
        'existe_acc': [],
        'average_pts_within_thresh': [],
    }

    existe_acc = np.mean((pre_existeded_binary == gt_existeded).astype(np.float32))
    metrics['existe_acc'].append(existe_acc)

    thresh_metrics = {
        'average_pts_within_thresh': [],
    }

    distances = np.sqrt(np.sum((gt_point - pre_point) ** 2, axis=-1))

    for thresh in pixel_thresholds:
        if gt_existeded == 1:
            pts_within_thresh = np.mean((distances <= thresh).astype(np.float32))
            thresh_metrics['average_pts_within_thresh'].append(pts_within_thresh)
        else:
            thresh_metrics['average_pts_within_thresh'].append(0)

    metrics['average_pts_within_thresh'] = np.mean(thresh_metrics['average_pts_within_thresh'])

    return metrics

def main(gt_point_folder_path, pre_point_folder_path, pre_existeded_folder_path):
    results = []
    num_gt_existeded = 0

    for subdir in os.listdir(gt_point_folder_path):
        subdir_path = os.path.join(gt_point_folder_path, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)

            for file in files:
                gt_point_path = os.path.join(subdir_path, file)
                pre_point_path = os.path.join(pre_point_folder_path, subdir, file)
                pre_existeded_path = os.path.join(pre_existeded_folder_path, subdir, file)

                gt_point = read_point_file(gt_point_path)
                pre_point = read_point_file(pre_point_path)
                pre_existeded = read_existed_score_file(pre_existeded_path)

                gt_existeded = int(np.linalg.norm(gt_point) > 0)

                if gt_existeded == 1:
                    num_gt_existeded += 1

                metrics = compute_metrics(gt_point, gt_existeded, pre_point, pre_existeded)
                results.append(metrics)

    return results, num_gt_existeded

if __name__ == "__main__":
    # gt_point_folder_path = r"/mnt/sdb3/home/siat-hci/ZhoZhangJun/CholecBP113-project/CholecBP113/test/videos-point"  # Update this path
    # pre_point_folder_path = r"/mnt/sdb3/home/siat-hci/ZhoZhangJun/CholecBP113-project/SAM2-Adapter-Video-MaskandPoint-oneBranch/saveResult/sam2video-adapter-onebranch/videos-point"  # Update this path
    # pre_existeded_folder_path = r"/mnt/sdb3/home/siat-hci/ZhoZhangJun/CholecBP113-project/SAM2-Adapter-Video-MaskandPoint-oneBranch/saveResult/sam2video-adapter-onebranch/videos-exited-score"  # Update this path

    # results, num_gt_existeded = main(gt_point_folder_path, pre_point_folder_path, pre_existeded_folder_path)
    
    # all_existe_acc = []
    # all_average_pts_within_thresh = []

    # for result in results:
    #     all_existe_acc.extend(result['existe_acc'])
    #     all_average_pts_within_thresh.append(result['average_pts_within_thresh'])

    # overall_existe_acc = np.mean(all_existe_acc)
    # overall_pts_within_thresh = np.sum(all_average_pts_within_thresh) / num_gt_existeded

    # print(num_gt_existeded)
    # print(f"Overall Existed Accuracy across all samples: {overall_existe_acc}")
    # print(f"Overall Average Points within Threshold across all samples: {overall_pts_within_thresh}")



    models = [7]
    
    # 循环遍历模型编号
    for model_num in models:
        print(f"Processing model-{model_num}")
        print("------------------------------------------------")

        gt_point_folder_path =      r"/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95/test/videos-point"  # Update this path
        pre_point_folder_path =     r'/home/pjl307/ZZJ/CholeBleedingPoint/SAM2.1_EdgeGenerator_Diandian/savel/mytest/model-{}/videos-point'.format(model_num)
        pre_existeded_folder_path = r'/home/pjl307/ZZJ/CholeBleedingPoint/SAM2.1_EdgeGenerator_Diandian/savel/mytest/model-{}/videos-exited-score'.format(model_num)

        results, num_gt_existeded = main(gt_point_folder_path, pre_point_folder_path, pre_existeded_folder_path)
        all_existe_acc = []
        all_average_pts_within_thresh = []
        for result in results:
            all_existe_acc.extend(result['existe_acc'])
            all_average_pts_within_thresh.append(result['average_pts_within_thresh'])
        overall_existe_acc = np.mean(all_existe_acc)
        overall_pts_within_thresh = np.sum(all_average_pts_within_thresh) / num_gt_existeded
        print(num_gt_existeded)
        print(f"Overall Existed Accuracy across all samples: {overall_existe_acc}")
        print(f"Overall Average Points within Threshold across all samples: {overall_pts_within_thresh}")