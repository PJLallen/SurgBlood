import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CholeBPTest(Dataset):
    def __init__(self, data_path, mode='test'):
        self.data_path = data_path
        self.mode = mode
        self.img_size = 512

        # Initialize transforms
        self.transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sample_lists = self.generate_video_list(data_path, mode)

    def __len__(self):
        return len(self.sample_lists)

    def __getitem__(self, index):
        video_info = self.sample_lists[index]
        name = video_info['name']

        video_path = os.path.join(self.data_path, self.mode, 'videos-image', name)
        mask_path  = os.path.join(self.data_path, self.mode, 'videos-mask', name)
        point_path = os.path.join(self.data_path, self.mode, 'videos-point', name)

        images = []
        masks = []
        points = []

        frame_names = sorted(f.split('.')[0] for f in os.listdir(video_path) if f.endswith('.jpg'))

        for frame_name in frame_names:
            img = Image.open(os.path.join(video_path, f'{frame_name}.jpg')).convert('RGB')
            mask = Image.open(os.path.join(mask_path, f'{frame_name}.jpg')).convert('L')
            point = self.load_point(point_path, frame_name)

            # Apply transformations
            img = self.transforms(img)
            mask = T.ToTensor()(T.Resize((self.img_size, self.img_size))(mask))

            # point = self.load_point(point_path, frame_name)

            images.append(img)  # Append image tensor
            masks.append(mask)  # Append mask tensor
            points.append(torch.tensor(point).float())  # Append point tensor

        video_meta_dict = {'filename_or_obj': name}


        return {
            'images': torch.stack(images),  # Stack list of tensors into one tensor
            'masks': torch.stack(masks),    # Stack list of tensors into one tensor
            'points': torch.stack(points),  # Stack point tensors
            'video_meta_dict': video_meta_dict,
            'frame_names': frame_names,
        }

    def generate_video_list(self, data_path, mode):
        sample_lists = []
        video_dir = os.path.join(data_path, mode, 'videos-image')
        video_list = sorted(os.listdir(video_dir))  # Sort video folders

        for name in video_list:
            sample_lists.append({'name': name})

        return sample_lists

    def load_point(self, point_path, frame_name):
        """Load the point information for a given frame."""
        point_file_path = os.path.join(point_path, f'{frame_name}.txt')
        point = (0, 0)  # Default point if not found

        if os.path.exists(point_file_path):
            try:
                with open(point_file_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        x, y = map(float, line.split())
                        point = (x, y)
                    else:
                        print(f"Warning: Empty point file for {frame_name}")
            except ValueError:
                print(f"Warning: Invalid point format in file {point_file_path}")
        else:
            print(f"Warning: Point file not found for {frame_name}")

        return point


# # Usage
# data_path = '/home/pjl307/ZZJ/CholeBleedingPoint/Cholec95-project/CholecBP95'
# test_dataset = CholeBPTest(data_path, mode='test')
# print("Total videos:", len(test_dataset))
