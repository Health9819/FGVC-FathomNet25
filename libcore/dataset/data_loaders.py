import os
import json
import scipy.io as scio
from cprint import cprint

__all__ = ['load_func_dict']


def cub_load_func(data_path, ):
    image_list = []
    with open(os.path.join(data_path, 'images.txt'), 'r') as f:
        for line in f.readlines():
            image_list.append(line.split(' ')[1].strip())

    train_test_split = []
    with open(os.path.join(data_path, 'train_test_split.txt'), 'r') as f:
        for line in f.readlines():
            train_test_split.append(int(line.split(' ')[1]))

    image_labels = []
    with open(os.path.join(data_path, 'image_class_labels.txt'), 'r') as f:
        for line in f.readlines():
            image_labels.append(int(line.split(' ')[1]) - 1)  # cub labels start from 1

    train_list = []
    valid_list = []
    for idx, is_train in enumerate(train_test_split):
        if is_train == 1:
            train_list.append((os.path.join(data_path, 'images', image_list[idx]), image_labels[idx]))
        else:
            valid_list.append((os.path.join(data_path, 'images', image_list[idx]), image_labels[idx]))

    return train_list, valid_list


def car_load_func(data_path, ):
    ret = []
    for phase in ['train', 'test']:
        devkit_path = 'devkit'
        annos = scio.loadmat(
            os.path.join(data_path, devkit_path, 'cars_%s_annos.mat' % phase)
        )['annotations'][0]  # np.arrays
        data_list = []
        for anno in annos:
            data_list.append((os.path.join(data_path, 'cars_' + phase, anno[-1][0]), int(anno[-2][0][0]) - 1))
        ret.append(data_list)
    # print(ret)
    return tuple(ret)


def dog_load_func(data_path, ):
    images_path = 'Images'
    ret = []
    for phase in ['train', 'test']:
        list_path = 'train_list.mat' if phase == 'train' else 'test_list.mat'
        annos = scio.loadmat(os.path.join(data_path, list_path))
        data_list = []
        for img, label in zip(annos['file_list'], annos['labels']):
            data_list.append((os.path.join(data_path, images_path, img[0][0]), int(label) - 1))
        ret.append(data_list)

    return tuple(ret)


def air_load_func(data_path, ):
    # cprint.info('name is not mapped with id in the data load function of FGVC-aircraft.')
    data_path = os.path.join(data_path, 'data')
    images_path = 'images'
    ret = []
    with open(os.path.join(data_path, 'variants.txt'), 'r') as f:
        label_names = list(f.readlines())
    label_names = {item.strip() : id for id, item in enumerate(label_names)} # fixed index
    for phase in ['trainval', 'test']:
        list_path = 'images_variant_' + phase + '.txt'

        data_list = []
        with open(os.path.join(data_path, list_path), 'r') as f:
            temp_list = list(f.readlines())
        for dd in temp_list:
            splits = dd.split(' ')
            img_name, lab_name = splits[0], ' '.join(splits[1:])
            gt_idx = label_names[lab_name.strip()]
            data_list.append((os.path.join(data_path, images_path, img_name + '.jpg'), gt_idx))

        ret.append(data_list)

    return tuple(ret)


def inat17_load_func(data_path, ):
    ret = []

    label_names = {}
    for phase in ['train', 'val']:
        list_path = phase + '2017.json'

        data_list = []
        with open(os.path.join(data_path, list_path), 'r') as f:
            json_file = json.load(f)
            img_list = json_file['images']
            lab_list = json_file['annotations']

        for dd in zip(img_list, lab_list):
            assert dd[0]['id'] == dd[1]['id']
            img_name, lab_name = dd[0]['file_name'], dd[1]['category_id']
            if lab_name not in label_names:
                label_names[lab_name] = len(list(label_names.keys()))
            gt_idx = label_names[lab_name]
            data_list.append((os.path.join(data_path, img_name), gt_idx))

        ret.append(data_list)

    return tuple(ret)

def nabird_load_func(data_path, ):

    class_list = []
    with open(os.path.join(data_path, 'classes.txt'), 'r') as f:
        for line in f.readlines():
            class_list.append(' '.join(line.split(' ')[1:]))

    image_list = []
    with open(os.path.join(data_path, 'images.txt'), 'r') as f:
        for line in f.readlines():
            image_list.append(line.split(' ')[1].strip())

    train_test_split = []
    with open(os.path.join(data_path, 'train_test_split.txt'), 'r') as f:
        for line in f.readlines():
            train_test_split.append(int(line.split(' ')[1]))

    image_labels = []
    with open(os.path.join(data_path, 'image_class_labels.txt'), 'r') as f:
        for line in f.readlines():
            image_labels.append(int(line.split(' ')[1]) - 1)

    used_label = list(set(image_labels))
    used_label = sorted(used_label)

    train_list = []
    valid_list = []
    for idx, is_train in enumerate(train_test_split):
        if is_train == 1:
            train_list.append((os.path.join(data_path, 'images', image_list[idx]), used_label.index(image_labels[idx])))
        else:
            valid_list.append((os.path.join(data_path, 'images', image_list[idx]), used_label.index(image_labels[idx])))
    return train_list, valid_list

def fathom_load_func(data_path, ):
    from torchvision.datasets import ImageFolder
    def get_img_label_list(data_root):
        """
        返回列表 [(img_path, label), ...]
        标签为数字（根据文件夹名自动映射）
        """
        dataset = ImageFolder(root=data_root)  # 仅用于获取标签映射，不加载数据
        class_to_idx = dataset.class_to_idx  # 获取类别到数字标签的映射

        img_label_list = []
        for class_name in os.listdir(data_root):
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                continue

            label = class_to_idx[class_name]  # 类别对应的数字标签
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_label_list.append((img_path, label))

        return img_label_list
    train_split, valid_split = get_img_label_list(os.path.join(data_path, "train")), get_img_label_list(os.path.join(data_path, "val"))

    return train_split + valid_split, valid_split

load_func_dict = {
    'cub': cub_load_func, 'dog': dog_load_func, 'car': car_load_func, 'air': air_load_func,
    'inat17': inat17_load_func, 'nabird': nabird_load_func, 'fathom': fathom_load_func
}
