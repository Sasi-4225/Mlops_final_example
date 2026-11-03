from prefect import flow, task
from train_model import train

@task(name="train-and-evaluate")
def train_task():
    return train()

@flow(name="heart-disease-ml-pipeline", log_prints=True)
def heart_disease_pipeline():
    metrics = train_task()
    print(f"Training completed: {metrics}")
    return metrics

if __name__ == "__main__":
    heart_disease_pipeline()
