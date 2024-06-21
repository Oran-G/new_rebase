import pandas as pd
import hydra
@hydra.main(config_path='../configs', config_name='defaults')
def main(cfg: DictConfig) -> None:
    df = pd.read_csv('cfg.io.final')
    filtered_df = df[df['bind'] == 'GATC'].head(20)
    print(filtered_df)

if __name__ == "__main__":
    main()
