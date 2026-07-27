import argparse
import model
import trainer_v2.trainer as trainer
import trainer_v2.dataset as dataset
import trainer_v2.load_latest_checkpoint as load_latest_checkpoint

if __name__=="__main__":
    argument_parser = argparse.ArgumentParser(description="Train model")
    argument_parser.add_argument("--version_name", type=str, required=True, help="Experiment/version name")
    argument_parser.add_argument("--model_name", type=str, required=True, help="Registered model name to train")
    argument_parser.add_argument("--dataset_name", type=str, required=True, help="Registered dataset name for training")
    argument_parser.add_argument("--trainer_name", type=str, required=True, help="Registered trainer name to use for training")
    argument_parser.add_argument("--resume", action="store_true", help="Whether to resume from the latest checkpoint for this model/dataset/trainer combination")

    args = argument_parser.parse_args()

    print(f"Training model '{args.model_name}' with version '{args.version_name}' on dataset '{args.dataset_name}' using trainer '{args.trainer_name}'")

    dataset = dataset.dataset(args.dataset_name)
    model = model.model(args.model_name, args.version_name)
    run_name = f"version={args.version_name}_dataset={args.dataset_name}_trainer={args.trainer_name}"
    trainer = trainer.trainer(args.trainer_name, args.model_name, run_name)

    latest_checkpoint = None
    if args.resume:
        latest_checkpoint = load_latest_checkpoint.load_latest_checkpoint(args.model_name, run_name)
        print(f"Resuming from checkpoint: {latest_checkpoint}")

    trainer.fit(model, dataset, ckpt_path=latest_checkpoint)