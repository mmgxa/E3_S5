from typing import Tuple, Dict

import lightning as L
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from lightning import LightningModule, LightningDataModule
import hydra
from omegaconf import DictConfig
from PIL import Image
import json

from dlearn import utils

log = utils.get_pylogger(__name__)


transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])



@utils.task_wrapper
def infer(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    #     L.seed_everything(cfg.seed, workers=True)

    # log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    # datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    # trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        # "datamodule": datamodule,
        "model": model,
        # "trainer": trainer,
    }

    log.info("Starting inference!")
    vit_model = model.model
    loaded_ckpt = torch.load(cfg.ckpt_path, map_location=torch.device('cpu'))

    my_model_kvpair=vit_model.state_dict()
    for key,value in loaded_ckpt['state_dict'].items():
        my_key = key[6:]
        my_model_kvpair[my_key] = loaded_ckpt['state_dict'][key]
    vit_model.load_state_dict(my_model_kvpair)
    vit_model.eval()

    img = Image.open(cfg.img_path)
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    labels = ['cat', 'dog']
    with torch.no_grad():
        prediction = vit_model(img_tensor)
    output = F.softmax(prediction, dim=1)
    torch.use_deterministic_algorithms(True)
    prediction_score, pred_label_idx = torch.topk(output, cfg.topk)
    result = {}
    for i in range(cfg.topk):
        result[labels[pred_label_idx.squeeze()[i].item()]] = round(prediction_score.squeeze()[i].item(),3)
    print(json.dumps(result))


    return result, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer.yaml")
def main(cfg: DictConfig):
    # test the model
    infer(cfg)


if __name__ == "__main__":
    main()
