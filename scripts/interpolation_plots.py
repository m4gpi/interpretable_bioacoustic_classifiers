import hydra
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def interpolation(cfg: DictConfig):
    data_module = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    ckpt = torch.load(cfg.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])

@hydra.main(
    version_base="1.3",
    config_path=str(rootutils.find_root() / "config"),
    config_name="eval.yaml"
)
def main(cfg: DictConfig):
    interpolation(cfg)

if __name__ == "__main__":
    main()
