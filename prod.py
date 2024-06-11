import wandb
import config
import argparse
import train
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    wandb.init( 
        project="Badminton App",
        sync_tensorboard=True,
        name=args.name,
        config={
            "epochs": config.epochs,
            "learning_rate": config.lr,
            "batch_size": config.batch_size,
            "num workers": config.num_workers,
            
            "classes": config.classes
        }
    )
    
    train.main()
    
    wandb.finish()

if __name__=='__main__':
    main()
