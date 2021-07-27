from src.approaches.train_audio2fl import Audio_Landmarks_block
from src.approaches.dataLoader import build_dataset

dataset = build_dataset(FREQ_SPLIT=4, batch_size=128)
model = Audio_Landmarks_block(dataset=dataset, loss='l2', epochs=200, model_save_dir='multi_task_l2_1', loss_weights=[1,5,1])
model.train(model_save=True)