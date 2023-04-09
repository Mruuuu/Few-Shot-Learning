"""
Reference: https://github.com/sicara/easy-few-shot-learning
"""

# package
import os
import pickle
import random

# torch
import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from torchvision import datasets, transforms


# Few shot learning dataset
class Fewshot(Dataset):
    
    def __init__(self, root, transform ,mode="train"):
        """
        Args:
            root: path for data directory 
            mode: 'train' / 'validation' / 'test'
        
        Dataset:
            data: dict, load from pickle file
                  data form: {"images": [], "labels": []}
        """
        
        # mode
        assert mode in ["train", "validation", "test"], "mode must be train / test / validation"
        self.mode = mode
        
        # img path
        self.root = root
        
        # transform
        self.transform = transform
        
        # load img & label
        with open(os.path.join(self.root, f"{mode}.pkl"), "rb") as f:
            self.data = pickle.load(f)
            # print("Total data num:", len(self.data["images"]))
        
        # debug
        if mode in ["train", "validation"]:
            assert len(self.data["images"]) == len(self.data["labels"]), "len(self.data['images']) != len(self.data['labels'])"
        else:
            assert len(self.data["sup_images"]) == len(self.data["sup_labels"]), "len(self.data['sup_images']) != len(self.data['sup_labels'])"
            assert len(self.data["sup_images"]) == len(self.data["qry_images"]), "len(self.data['sup_images']) != len(self.data['qry_images'])"


    def __len__(self):
        if self.mode in ["train", "validation"]:
            return len(self.data["images"])
        else:
            return len(self.data["sup_images"])


    def __getitem__(self, index):
        
        if self.mode == 'test':
            sup_img = torch.tensor(self.data["sup_images"][index])
            sup_label = torch.tensor(self.data["sup_labels"][index])
            qry_img = torch.tensor(self.data["qry_images"][index])
            return sup_img, sup_label, qry_img
        else:
            img = torch.tensor(self.data["images"][index])
            img = self.transform(img)
            return img, self.data["labels"][index]
    
    def get_labels(self):
        
        assert self.mode in ["train", "validation"], "function get_labels only valid for train / validation"
        return self.data["labels"][:]

        
class Fewshot_batch_sampler(Sampler):
    
    def __init__(self, dataset: Fewshot, n_way: int, n_shot: int, n_query: int, n_tasks: int):
        """
        Args:
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
            root: path for data directory 
            mode: 'train' / 'val' / 'test'
        
        Dataset:
            data: dict, load from pickle file
                  data form: {"images": [], "labels": []}
            ---------------------------- after transfer ----------------------------
            label_items_dict: {0: [imgs with label 0], 1: [imgs with label 1], ...}
        """
        super().__init__(data_source=None)
        
        # param
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_tasks = n_tasks
        
        # items_label_dict include all the item in the same labels
        self.items_label_dict = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_label_dict:
                self.items_label_dict[label].append(item)
            else:
                self.items_label_dict[label] = [item]

        
    def __len__(self):
        
        return self.n_tasks


    def __iter__(self):
        
        # generate n_tasks fsl task by randomly select img with the same label
        for _ in range(self.n_tasks):
            yield torch.cat([
                    torch.tensor(random.sample(self.items_label_dict[label], self.n_shot + self.n_query))
                    for label in random.sample(self.items_label_dict.keys(), self.n_way)
                ]).tolist()
    

    def collate_fn(self, input_data):

        # true class
        class_ids = list({x[1] for x in input_data})

        # merge all imgs & reshape
        imgs = torch.cat([x[0].unsqueeze(0) for x in input_data])
        imgs = imgs.reshape((self.n_way, self.n_shot + self.n_query, *imgs.shape[1:]))
        labels = torch.tensor([class_ids.index(x[1]) for x in input_data])
        labels = labels.reshape((self.n_way, self.n_shot + self.n_query))

        # support & query
        support_imgs = imgs[:, :self.n_shot].reshape((-1, *imgs.shape[2:]))
        query_imgs = imgs[:, self.n_shot:].reshape((-1, *imgs.shape[2:]))
        support_labels = labels[:, :self.n_shot].flatten()
        query_labels = labels[:, self.n_shot:].flatten()

        return support_imgs, support_labels, query_imgs, query_labels, class_ids


# default transform
def set_transform():

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    return transform


# dataloader for episodic fsl, (use sup/qry set while training & testing)
def set_episodic_data_loader(root, transform, mode, n_way, n_shot, n_query, n_task_train, num_workers):
            
    dataset = Fewshot(root=root, transform=transform, mode=mode)
    sampler = Fewshot_batch_sampler(dataset, n_way=n_way, n_shot=n_shot, n_query=n_query, n_tasks=n_task_train)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                        num_workers=num_workers, pin_memory=True,
                                        collate_fn=sampler.collate_fn)
    return dataset, sampler, loader


# dataloader for classical fsl, (train under supervised learning)
def set_classical_data_loader(root, transform, batch_size, num_workers):
    
    dataset = Fewshot(root=root, transform=transform, mode="train")
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                        num_workers=num_workers, pin_memory=True,
                                        shuffle=True)
    
    return dataset, loader