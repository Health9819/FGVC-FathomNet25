import json
from libcore.config import get_cfg
from pprint import pprint
import pytorch_lightning as pl
import torch.cuda
from easydict import EasyDict as edict

from libcore.dataset import build_dataset, build_dataloader, build_transform
from libcore.models import build_model
from torchvision.datasets import ImageFolder

def valid_net(args):
    cfg, updater = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    pprint(cfg)

    torch.cuda.empty_cache()
    data_cfg = cfg.DATA
    _, valid_dataset, cls_num = build_dataset(
        data_cfg.NAME.lower(), data_cfg.PATH, data_cfg.DATASET, build_transform(data_cfg.INPUT),
    )
    valid_dataloader = build_dataloader(valid_dataset, data_cfg.DATALOADER, shuffle=False)

    fgic_model = build_model(cfg.MODEL, cls_num)
    fgic_model.load_state_dict(
        torch.load(cfg['best_path'], map_location='cpu')['state_dict'], strict=False,
    )

    # build network
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=cfg.TRAINER.DEVICE_ID,
        num_nodes=1,
        max_epochs=1,
    )
    trainer.validate(fgic_model, valid_dataloader, verbose=True)


def valid_fathom(args):
    import torchvision.transforms as torchtfs
    import os
    from PIL import Image
    import torch
    from torch.utils.data import Dataset, DataLoader

    class ImageDataset(Dataset):
        def __init__(self, img_dir, transform=None):
            self.img_dir = img_dir
            self.img_paths = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            self.transform = transform

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_paths[idx])
            image = Image.open(img_path).convert('RGB')  # 确保转为RGB格式
            if self.transform:
                image = self.transform(image)
            return image, img_path

    cfg, updater = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    pprint(cfg)

    fgic_model = build_model(cfg.MODEL, 79)
    fgic_model.load_state_dict(
        torch.load(cfg['best_path'], map_location='cpu', weights_only=False)['state_dict'], strict=False,
    )
    network = fgic_model.network
    network.to('cuda:0')

    ROOT_PATH = '/root/autodl-tmp/dataset'

    dataset = ImageFolder(root=os.path.join(ROOT_PATH, 'train'))  # 仅用于获取标签映射，不加载数据
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print(idx_to_class)
    valid_trans = torchtfs.Compose(
        [
            torchtfs.Resize(512),
            torchtfs.Pad(32),
            torchtfs.TenCrop(448),
            torchtfs.Lambda(lambda crops: torch.stack([torchtfs.ToTensor()(crop) for crop in crops])),
            torchtfs.Lambda(lambda crops: torch.stack([torchtfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops]))
        ]
    )

    dataset = ImageDataset(os.path.join(ROOT_PATH, 'rois'), transform=valid_trans)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=False)
    network.train()
    with torch.no_grad():
        for ims, ips in dataloader:
            ims = ims.to('cuda:0')
            b, n, c, h, w = ims.shape
            network(ims.reshape(b * n, c, h, w))

    network.eval()
    dataset = ImageDataset(os.path.join(ROOT_PATH, 'rois'), transform=valid_trans)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    logits, im_paths = [], []
    with torch.inference_mode():
        for ims, ips in dataloader:
            ims = ims.to('cuda:0')
            b, n, c, h, w = ims.shape
            p1, p2, p3, pc = network(ims.reshape(b * n, c, h, w))
            p = p2 + p3
            p = p.reshape(b, n, -1).mean(1)
            logits.append(p.detach().clone())
            im_paths.extend(ips)
    logits = torch.cat(logits)
    p_names = [idx_to_class[item.item()] for item in logits.argmax(-1).reshape(-1)]

    torch.save(
        {
            'logits': logits, 'p_names': p_names, 'im_paths': im_paths, 'id_to_cls': idx_to_class,
        }, 'results_v4.pkl'
    )



