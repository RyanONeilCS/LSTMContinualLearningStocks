# LSTM Stock Prediction from Scratch

Ryan O'Neil — Rco220002

A Long Short-Term Memory network written by hand in NumPy and used to predict
next day movement for 12 S&P 500 companies. No deep learning libraries are
used anywhere. The cell, the forward pass, backpropagation through time and
the training loop are all implemented directly.

## Requirements

    pip install -r requirements.txt

numpy, pandas, matplotlib, requests. Python 3.9 or newer.

## Running it

    python main.py

The first run needs internet. It downloads 12 CSVs from

    https://github.com/RyanONeilCS/MachineLearningProjectData

and caches them in `data_cache/`. Every run after that reads the cache and
needs no network. The dataset is hosted rather than shipped because it is too
large to include with the submission.

Full run takes a few minutes. The hyperparameter sweep is fast, the
incremental experiment is the slow part.

## Checking the gradients

    python test_gradients.py

Compares every analytical gradient from the backward pass against a central
finite difference estimate. Expected output ends with

    WORST 3.255e-07 -> PASS

This is the verification quoted in Section III of the report.

## Switching the target

The target is set in one place, at the top of `data.py`

    TARGET = "daily_range"     # tomorrow's volatility
    TARGET = "daily_return"    # tomorrow's price move

`main.py` reads `data.TARGET` into its config, so the experiment log can never
disagree with what the code actually did. Change that one line and rerun to
reproduce the controlled comparison in Section V.

## Files

    lstm.py             the algorithm, all of it by hand
                        sigmoid, parameters, lstm_step, forward, backward, train

    data.py             download, caching, features, chronological split,
                        scaling and windowing

    main.py             config, the 11 run hyperparameter sweep, the
                        incremental experiment, evaluation and figures

    test_gradients.py   numerical gradient check

    log.txt             every experiment, appended as a numbered block with
                        its full config and results

## Output

Running `main.py` appends to `log.txt` and writes four figures

    loss_curve.png         training loss for the base run
    sweep_skill.png        skill of every hyperparameter setting
    forgetting_matrix.png  skill per company after each incremental stage
    skill_vs_stages.png    skill as companies are added one at a time

## Reading the results

Skill is the model's RMSE divided by the RMSE of a baseline that always
predicts the training set mean of the target. Below 1.0 beats the baseline,
1.0 means no improvement, above 1.0 is worse.

Headline numbers

    next day returns      skill 1.0002    R2 -0.003
    next day volatility   skill 0.8691    R2  0.2326

Same model, same data, same hyperparameters. The only difference is which
column becomes the target.

## Notes

`data_cache/` is created on first run and is not part of the submission.
