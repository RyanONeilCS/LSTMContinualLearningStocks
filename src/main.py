"""
Entry point for the LSTM stock-forecasting experiments.

Run from the repo root with:
    python src/main.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

import data
from lstm import parameters, forward, train

#Outputs live at the repo root so they are found no matter where this is run from
LOG_PATH = os.path.join(data.ROOT, "results", "log.txt")
FIG_DIR = os.path.join(data.ROOT, "figures")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok = True)
os.makedirs(FIG_DIR, exist_ok = True)

#Everything that changes a result goes here so the whole dict can be logged
CONFIG = {
    # data
    "tickers": data.TICKERS,
    "target": data.TARGET,
    "n_features": 3,
    "window": 60,
    "stride": 1,
    "split": .8,

    # model
    "hidden_size": 32,
    "seed": 0,

    # training
    "learning_rate": .1,
    "epochs": 30,
    "batch_size": 64,
    "clip": 5.0,

    # incremental experiment
    "stage_epochs": 15,
}

#Each run changes one setting away from CONFIG and nothing else
#Delete a line to skip that run
RUNS = [
    ("base run", {}),
    ("smaller model", {"hidden_size": 16}),
    ("bigger model", {"hidden_size": 64}),
    ("short window", {"window": 20}),
    ("long window", {"window": 120}),
    ("small learning rate", {"learning_rate": .01}),
    ("large learning rate", {"learning_rate": .5}),
    ("fewer epochs", {"epochs": 10}),
    ("more epochs", {"epochs": 60}),
    ("bigger batches", {"batch_size": 128}),
    ("different seed", {"seed": 7}),
]


#Score predictions against the baseline that guesses the training mean
#skill = rmse / baseline rmse so under 1 is good
#direction is vs the mean not zero since range is always positive
#pred_std near zero means the model gave up and is repeating the mean
def evaluate(y_pred: np.ndarray, y_true: np.ndarray, train_mean: float):

    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    #Root mean squared error in the same units as the target
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

    #Mean absolute error so one crash day cannot dominate
    mae = np.mean(np.abs(y_pred - y_true))

    #Same two numbers for the baseline
    baseline_rmse = np.sqrt(np.mean((y_true - train_mean) ** 2))
    baseline_mae = np.mean(np.abs(y_true - train_mean))

    #Skill is the error as a fraction of the baseline
    skill = rmse / baseline_rmse
    skill_mae = mae / baseline_mae

    #R squared goes negative when the model is worse than the average
    ss_residual = np.sum((y_true - y_pred) ** 2)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    #Correlation of prediction and truth
    #A constant prediction has no spread so call it 0
    if np.std(y_pred) < 1e-12:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(y_pred, y_true)[0, 1])

    #Did it call a bigger or smaller than usual day
    #Baseline is always guessing whichever side is more common
    direction = np.mean(np.sign(y_pred - train_mean) == np.sign(y_true - train_mean))
    above_rate = np.mean(y_true > train_mean)
    baseline_direction = max(above_rate, 1 - above_rate)

    #Near zero means the model gave up
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)

    return {
        "rmse": round(float(rmse), 6),
        "mae": round(float(mae), 6),
        "baseline_rmse": round(float(baseline_rmse), 6),
        "baseline_mae": round(float(baseline_mae), 6),
        "skill": round(float(skill), 4),
        "skill_mae": round(float(skill_mae), 4),
        "r2": round(float(r2), 4),
        "correlation": round(float(correlation), 4),
        "direction": round(float(direction), 4),
        "baseline_direction": round(float(baseline_direction), 4),
        "pred_std": round(float(pred_std), 6),
        "true_std": round(float(true_std), 6),
        "n_test": int(len(y_true)),
    }


#Append one run to the log as a numbered block
#Number carries on from whatever is already in the file
def log_run(config: dict, results: dict, note: str = "", path: str = LOG_PATH):

    number = 1
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as file:
            number = sum(1 for line in file if line.startswith("EXPERIMENT ")) + 1

    with open(path, "a", encoding = "utf-8") as file:
        file.write(f"EXPERIMENT {number}\n")
        if note:
            file.write(f"  Note: {note}\n")

        file.write("  Parameters:\n")
        for key, value in config.items():
            if key == "tickers":
                value = f"{len(value)} stocks ({', '.join(value)})"
            file.write(f"    {key} = {value}\n")

        file.write("  Results:\n")
        for key, value in results.items():
            file.write(f"    {key} = {value}\n")

        file.write("\n")

    return number


#Experiment 1 train the model once per hyperparameter setting
#Fills the experiment log
sweep_results = []

for note, changes in RUNS:

    #Start from CONFIG and change only this run's setting
    run_config = dict(CONFIG)
    run_config.update(changes)

    #Reloaded every run since window changes how much data there is
    X_train, y_train, X_test, y_test = data.load(run_config["tickers"], run_config["window"], run_config["stride"], run_config["split"], verbose = False)

    #Train the model
    params = parameters(run_config["n_features"], run_config["hidden_size"], run_config["seed"])
    history = train(X_train, y_train, params, run_config["learning_rate"], run_config["epochs"], run_config["batch_size"], run_config["clip"], run_config["seed"])

    #Predict on the test set with the model frozen
    y_pred, _, _ = forward(X_test, params)

    #Evaluate and write the run to the log
    results = evaluate(y_pred, y_test, np.mean(y_train))
    results["train_windows"] = int(len(y_train))
    results["final_train_mse"] = round(float(history[-1]), 8)

    number = log_run(run_config, results, note)
    sweep_results.append((note, results, history))

    print(f"experiment {number:2d} {note:22s} skill {results['skill']:.4f} r2 {results['r2']:+.4f} corr {results['correlation']:+.4f} dir {results['direction']:.4f}")

#Base run is first and is the setup the report quotes
base_history = sweep_results[0][2]


#Experiment 2 one model that learns the companies one at a time
#Initialised once then warm started through every stage and never reset
#Tested on all 12 after each stage which gives the forgetting matrix

#Keep every company's data separate so each one can be tested on its own
per_ticker = data.load_each(CONFIG["tickers"], CONFIG["window"], CONFIG["stride"], CONFIG["split"], verbose = False)

#Each company needs its own baseline since their sizes differ
train_means = {t: np.mean(per_ticker[t][1]) for t in CONFIG["tickers"]}

#One fresh set of parameters and this is the only initialisation
incremental_params = parameters(CONFIG["n_features"], CONFIG["hidden_size"], CONFIG["seed"])

#Each row is one stage and each column is one company's test set
skill_matrix = []

for stage, ticker in enumerate(CONFIG["tickers"]):

    #Train on this company only and carry the last stage's weights forward
    X_stage, y_stage = per_ticker[ticker][0], per_ticker[ticker][1]
    train(X_stage, y_stage, incremental_params, CONFIG["learning_rate"], CONFIG["stage_epochs"], CONFIG["batch_size"], CONFIG["clip"], CONFIG["seed"])

    #Test on every company including ones not shown yet
    row = []
    for other in CONFIG["tickers"]:
        X_other, y_other = per_ticker[other][2], per_ticker[other][3]
        y_other_pred, _, _ = forward(X_other, incremental_params)
        other_results = evaluate(y_other_pred, y_other, train_means[other])
        row.append(other_results["skill"])

    skill_matrix.append(row)

    #Compare companies seen so far against unseen ones
    seen = np.mean(row[:stage + 1])
    unseen = np.mean(row[stage + 1:]) if stage + 1 < len(CONFIG["tickers"]) else float("nan")
    print(f"stage {stage + 1:2d} trained on {ticker:5s} seen {seen:.4f} unseen {unseen:.4f}")

skill_matrix = np.array(skill_matrix)

#Print the forgetting matrix with stages as rows and companies as columns
print("       " + "".join(f"{t:>7s}" for t in CONFIG["tickers"]))
for stage, ticker in enumerate(CONFIG["tickers"]):
    print(f"{ticker:5s}  " + "".join(f"{v:7.3f}" for v in skill_matrix[stage]))

#First company has the most stages to be forgotten across
first_ticker = CONFIG["tickers"][0]
first_after_learning = skill_matrix[0, 0]
first_at_end = skill_matrix[-1, 0]

print(f"{first_ticker} skill after its own stage {first_after_learning:.4f}")
print(f"{first_ticker} skill after all stages {first_at_end:.4f}")
print(f"forgetting, positive means worse {first_at_end - first_after_learning:+.4f}")

incremental_results = {
    "stages": len(CONFIG["tickers"]),
    "final_mean_skill": round(float(skill_matrix[-1].mean()), 4),
    "worst_stage": CONFIG["tickers"][int(np.argmax(skill_matrix.mean(axis = 1)))],
    "worst_stage_mean_skill": round(float(skill_matrix.mean(axis = 1).max()), 4),
    "first_ticker": first_ticker,
    "first_skill_after_own_stage": round(float(first_after_learning), 4),
    "first_skill_after_all_stages": round(float(first_at_end), 4),
    "forgetting": round(float(first_at_end - first_after_learning), 4),
}

log_run(CONFIG, incremental_results, "incremental model, one company per stage")

#Matrix is too wide for the block format so it goes underneath
with open(LOG_PATH, "a", encoding = "utf-8") as f:
    f.write("  Skill matrix, rows are stages and columns are companies\n")
    f.write("       " + "".join(f"{t:>8s}" for t in CONFIG["tickers"]) + "\n")
    for stage, ticker in enumerate(CONFIG["tickers"]):
        f.write(f"  {ticker:5s}" + "".join(f"{v:8.3f}" for v in skill_matrix[stage]) + "\n")
    f.write("\n")


#Figure 1 training loss
plt.figure()
plt.plot(base_history)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training loss, base run")
plt.savefig(os.path.join(FIG_DIR, "loss_curve.png"), dpi = 150, bbox_inches = "tight")
plt.close()

#Figure 2 skill of every setting
labels = [note for note, _, _ in sweep_results]
skills = [results["skill"] for _, results, _ in sweep_results]

plt.figure(figsize = (10, 5))
plt.bar(range(len(skills)), skills, color = "steelblue")
plt.axhline(1.0, linestyle = "--", color = "red", label = "baseline, anything above this loses")
plt.xticks(range(len(labels)), labels, rotation = 45, ha = "right")
plt.ylabel("Skill ratio (lower is better)")
plt.title("Skill across every hyperparameter setting")
plt.legend()
plt.savefig(os.path.join(FIG_DIR, "sweep_skill.png"), dpi = 150, bbox_inches = "tight")
plt.close()

#Figure 3 forgetting matrix heatmap
plt.figure(figsize = (8, 6))
plt.imshow(skill_matrix, cmap = "RdYlGn_r", aspect = "auto")
plt.colorbar(label = "Skill ratio (lower is better)")
plt.xticks(range(len(CONFIG["tickers"])), CONFIG["tickers"], rotation = 45)
plt.yticks(range(len(CONFIG["tickers"])), [f"after {t}" for t in CONFIG["tickers"]])
plt.xlabel("Tested on")
plt.ylabel("Training stage")
plt.title("Skill ratio per company after each incremental stage")
plt.savefig(os.path.join(FIG_DIR, "forgetting_matrix.png"), dpi = 150, bbox_inches = "tight")
plt.close()

#Figure 4 does skill drift as companies are added
stages = range(1, len(CONFIG["tickers"]) + 1)
plt.figure()
plt.plot(stages, skill_matrix.mean(axis = 1), marker = "o", label = "mean skill, all companies")
plt.plot(stages, skill_matrix[:, 0], marker = "s", label = f"skill on {first_ticker}, the first company")
plt.axhline(1.0, linestyle = "--", color = "grey", label = "baseline")
plt.xlabel("Companies trained on so far")
plt.ylabel("Skill ratio (lower is better)")
plt.title("Skill as companies are added one at a time")
plt.legend()
plt.savefig(os.path.join(FIG_DIR, "skill_vs_stages.png"), dpi = 150, bbox_inches = "tight")
plt.close()

print("saved to figures/ loss_curve.png, sweep_skill.png, forgetting_matrix.png, skill_vs_stages.png")
