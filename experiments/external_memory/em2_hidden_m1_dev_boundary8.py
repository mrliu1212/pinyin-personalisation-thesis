from experiments.external_memory import em2_hidden_m1_dev as experiment

# Pre-registered single upper-bound check.
experiment.LAMBDAS = (*experiment.LAMBDAS, 8.0)

experiment.main()
