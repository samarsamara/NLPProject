import wandb
YOUR_WANDB_USERNAME = "rwanbd"
project = "Strategy_Transfer_TACL"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "TransformerLSTM: seeds",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [True]},
        "architecture": {"values": ["TransformerLSTM"]},
        "seed":{"values":[40,42,58,60,72,90,80,70,72,84,120,92,99,98,100,102,104,105,106,108,110,112,114,116,118,126,122,124,128,130,132,134,136,138,140,142,144,146,148,150,152,154,156,158,160,162,164,168,170,172,174]}   
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
