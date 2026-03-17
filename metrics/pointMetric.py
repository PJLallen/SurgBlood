import numpy as np
import os

def read_point_file(file_path):
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
    
    #gt_point
    gt_point=gt_point*512
    #pre_point:
    pre_point=pre_point*256


    # pixel_thresholds = [1, 2, 4, 8, 16]  # Corresponding to 256x256 images
    pixel_thresholds = [10, 20, 40, 80]  # Corresponding to 256x256 images

    #
    pre_existeded_binary = np.array(pre_existeded > 0.5, dtype=np.int32)

    metrics = {
        'existe_acc': [],
        'average_pts_within_thresh': [],
        'average_jaccard': []
    }


    existe_acc = np.mean((pre_existeded_binary == gt_existeded).astype(np.float32))
    metrics['existe_acc'].append(existe_acc)

    thresh_metrics = {
        'average_pts_within_thresh': [],
        'average_jaccard': []
    }

    distances = np.sqrt(np.sum((gt_point - pre_point) ** 2, axis=-1))
    # print(distances)
    # distances = 0

    for thresh in pixel_thresholds:

        # print(gt_existeded)
        # input()

        if gt_existeded == 1:
            # print(1111111111111)
            # input()
            # distances = np.sqrt(np.sum((gt_point - pre_point) ** 2, axis=-1))
            # print(distances)
            
            pts_within_thresh = np.mean((distances <= thresh).astype(np.float32))#计算距离小于阈值的点的比例
            # print(pts_within_thresh)
            thresh_metrics['average_pts_within_thresh'].append(pts_within_thresh)
        else:
            thresh_metrics['average_pts_within_thresh'].append(0)

        # true_positives = np.sum((distances <= thresh) & (gt_existeded == 1) & (pre_existeded_binary == 1))#真实存在，预测存在,且距离小于阈值

        # false_positives = np.sum((pre_existeded_binary == 1) & (gt_existeded == 0) | ((distances > thresh) & (pre_existeded_binary == 1)))

        # false_negatives = np.sum((gt_existeded == 1) & ((pre_existeded_binary == 0) | (distances > thresh)))#真实存在，预测不存在，或者距离大于阈值

        # jaccard = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0
        # metrics['average_jaccard'].append(jaccard)

    metrics['average_pts_within_thresh'] = np.mean(thresh_metrics['average_pts_within_thresh'])
    metrics['average_jaccard'] = np.mean(thresh_metrics['average_jaccard'])

    return metrics

def main(gt_point_folder_path, pre_point_folder_path, pre_existeded_folder_path):
    files = os.listdir(gt_point_folder_path)
    results = []
    num_gt_existeded=0

    for file in files:
        gt_point_path = os.path.join(gt_point_folder_path, file)
        pre_point_path = os.path.join(pre_point_folder_path, file)
        pre_existeded_path = os.path.join(pre_existeded_folder_path, file)

        gt_point = read_point_file(gt_point_path)
        # print(gt_point)
        pre_point = read_point_file(pre_point_path)
        # print(pre_point)
        pre_existeded = read_existed_score_file(pre_existeded_path)
        # print(pre_existeded)

        #
        gt_existeded = int(np.linalg.norm(gt_point) > 0)
        # print(gt_existeded)


        if gt_existeded==1:
            num_gt_existeded+=1

        metrics = compute_metrics(gt_point, gt_existeded, pre_point, pre_existeded)
        results.append(metrics)

    return results,num_gt_existeded

if __name__ == "__main__":
    gt_point_folder_path = r"/home/siat-hci/ZhoZhangJun/CholeBP51-imageLevel-Project/CholeBP51-image-level/test/point"
    pre_point_folder_path = r"/home/siat-hci/ZhoZhangJun/CholeBP51-imageLevel-Project/SAM2-Adapter-mask-point/saveResult/weightSAM2-Adapter-hiera-l-point-512-unfreezeeMemory/point"
    pre_existeded_folder_path = r"/home/siat-hci/ZhoZhangJun/CholeBP51-imageLevel-Project/SAM2-Adapter-mask-point/saveResult/weightSAM2-Adapter-hiera-l-point-512-unfreezeeMemory/exited_score"

    results,num_gt_existeded = main(gt_point_folder_path, pre_point_folder_path, pre_existeded_folder_path)
    
    # Lists to collect all values for the three metrics
    all_existe_acc = []
    all_average_pts_within_thresh = []
    all_average_jaccard = []

    for result in results:

        # print(f"Existed Accuracy: {result['existe_acc']}")
        # print(f"Average Points within Threshold: {result['average_pts_within_thresh']}")
        # print(f"Average Jaccard: {result['average_jaccard']}")
        # input()

        all_existe_acc.extend(result['existe_acc'])  # Append all accuracy values
        all_average_pts_within_thresh.append(result['average_pts_within_thresh'])  # Append average points within threshold
        all_average_jaccard.append(result['average_jaccard'])  # Append average Jaccard

    # Calculate the overall metrics by averaging across all samples
    overall_existe_acc = np.mean(all_existe_acc)



    print(num_gt_existeded)
    overall_pts_within_thresh = np.sum(all_average_pts_within_thresh) / num_gt_existeded
    # overall_pts_within_thresh = np.mean(all_average_pts_within_thresh) 

    overall_jaccard = np.mean(all_average_jaccard)

    # Print the overall metrics
    print(f"Overall Existed Accuracy across all samples: {overall_existe_acc}")
    print(f"Overall Average Points within Threshold across all samples: {overall_pts_within_thresh}")
    print(f"Overall Average Jaccard across all samples: {overall_jaccard}")



# Overall Existed Accuracy across all samples: 0.723022997379303
# Overall Average Points within Threshold across all samples: 0.37654852867126465
# Overall Average Jaccard across all samples: 0.0014803272302298404
    




