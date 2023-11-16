# Train
for seed in 22 94 95 96 99 3407
do
    for model in lightgbm xgboost catboost tabnet
    do
        python src/train.py models=$model models.seed=$seed
    done
done

# Inference
for seed in 22 94 95 96 99 3407
do
    for model in lightgbm xgboost catboost tabnet
    do
        python src/predict.py models=$model models.seed=$seed
    done
done