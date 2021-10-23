# Some dvc commands

    dvc run exp -S model.units1=30 -S model.optimizer=adam
    dvc exp show

    dvc plots show -o "plots/accuracy" -y accuracy