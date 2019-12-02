import os
import numpy as np
import h5py
import torch
import torch.utils.data as data
import pickle
from PIL import Image


class CreateData(data.Dataset):
    def __init__(self, dataset_dict):
        self.len_dset_dict = len(dataset_dict)
        self.rgb = dataset_dict['rgb']
        self.depth = dataset_dict['depth']
        self.seg_label = dataset_dict['seg_label']

        if self.len_dset_dict > 3:
            self.class_label = dataset_dict['class_label']
            self.use_class = True

    def __getitem__(self, index):
        rgb_img = self.rgb[index]
        depth_img = self.depth[index]
        seg_label = self.seg_label[index]

        rgb_img = torch.from_numpy(rgb_img)
        depth_img = torch.from_numpy(depth_img)

        dataset_list = [rgb_img, depth_img, seg_label]

        if self.len_dset_dict > 3:
            class_label = self.class_label[index]
            dataset_list.append(class_label)
        return dataset_list

    def __len__(self):
        return len(self.seg_label)


def get_data(opt, dset_info, use_train=True, use_test=True, visualize = False):
    """
    Load NYU_v2 or SUN rgb-d dataset in hdf5 format from disk and prepare
    it for classifiers.
    """
    if list(dset_info.keys())[0] == "NYU":
        # Load the chosen datasets path
        if os.path.exists(opt.dataroot):
            path = opt.dataroot
        else:
            raise Exception('Wrong datasets requested. Please choose either "NYU" or "SUN"')

        h5file = h5py.File(path, 'r')
    elif list(dset_info.keys())[0] == "SUN":
        h5file = None
        
    train_dataset_generator = None
    test_dataset_generator = None

    # Create python dicts containing numpy arrays of training samples
    if use_train:
        train_dataset_generator = dataset_generator(h5file, 'train', opt, dset_info, visualize)
        print('[INFO] Training set generator has been created')

    # Create python dicts containing numpy arrays of test samples
    if use_test:
        test_dataset_generator = dataset_generator(h5file, 'test', opt, dset_info, visualize)
        print('[INFO] Test set generator has been created')
    
    if h5file is not None:
        h5file.close()
    return train_dataset_generator, test_dataset_generator


def dataset_generator(h5file, dset_type, opt, dset_info, visualize):
    """
    Move h5 dictionary contents to python dict as numpy arrays and create dataset generator
    """
    use_class = opt.use_class
    
    if list(dset_info.keys())[0] == "NYU":
        dataset_dict = dict()
        # Create numpy arrays of given samples
        dataset_dict['rgb'] = np.array(h5file['rgb_' + dset_type],  dtype=np.float32)
        dataset_dict['depth'] = np.array(h5file['depth_' + dset_type], dtype=np.float32)
        dataset_dict['seg_label'] = np.array(h5file['label_' + dset_type], dtype=np.int64)

        # If classification loss is included in training add the classification labels to the dataset as well
        if use_class:
            dataset_dict['class_label'] = np.array(h5file['class_' + dset_type], dtype=np.int64)
            
        print(dataset_dict['rgb'].shape)
        print(dataset_dict['depth'].shape)
        print(dataset_dict['seg_label'].shape)
        return CreateData(dataset_dict)

    elif list(dset_info.keys())[0] == "SUN":
        root = opt.dataroot
        splits = pickle.load(open(os.path.join(root, "splits.pkl"), "rb"), encoding="latin1")
        tsplits = None
        if dset_type == 'train':
            tsplits = np.arange(2000, 9001) #1 - 9001
        elif dset_type== 'test':
            tsplits = np.arange(9001, 10335) #9001 - 10335

        rgb = []
        depth = []
        mask = []
        for index in tsplits:
            rimg = np.array(Image.open(os.path.join(root, "images-224", str(index)+".png")))

            rimg = rimg.transpose(2, 0, 1)
            rgb.append(rimg)
            
            dimg = np.array(Image.open(os.path.join(root, "depth-inpaint-u8-224", str(index)+".png")))
            dimg = dimg[:, :, np.newaxis]
            dimg = dimg.transpose(2, 0, 1)
            depth.append(dimg)            
            mask.append(np.array(Image.open(os.path.join(root, "seglabel-224", str(index)+".png"))))

            if visualize:
                if index == tsplits[0] + 300:    # only 20 images for visualization, limited memory while training                
                    break
            
        dataset_dict = dict()
        dataset_dict['rgb'] = np.array(rgb, dtype=np.float32)
        dataset_dict['depth'] = np.array(depth, dtype=np.float32)
        dataset_dict['seg_label'] = np.array(mask, dtype=np.int64)
        
        print(dataset_dict['rgb'].shape)
        #print(dataset_dict['depth'].shape)
        #print(dataset_dict['seg_label'].shape)
        
        return CreateData(dataset_dict)
