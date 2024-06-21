import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
@hydra.main(config_path='../configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    df = pd.read_csv(cfg.io.final)
    filtered_df = df[df['bind'] == 'GATC'].head(20)
    print(filtered_df.to_string())

if __name__ == "__main__":
    main()
