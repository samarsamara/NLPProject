import wandb
YOUR_WANDB_USERNAME = "s-samar2499"
project = "Strategy_Transfer_TACL_s-samar2499"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTMTransformer: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [True]},
        "architecture": {"values": ["LSTMTranformer"]},
        "seed": {"values": list(range(1, 6))},
        "online_simulation_factor": {"values": [0, 4]},
        "features": {"values": ["EFs", "GPT4", "BERT"]},
        "ENV_LEARNING_RATE":{"values":[0.001,0.002,0.0025,0.005]}   
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {ssamar2499}/{project}/{sweep_id}")
