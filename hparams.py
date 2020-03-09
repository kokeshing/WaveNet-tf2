
upsample_scales = [4, 8, 8]

seq_len = 10240
sampling_rate = 24000
num_mels = 80
n_fft = 1024
hop_size = 256
win_size = 1024

learning_rate = 1e-3
beta_1 = 0.9
exponential_decay_rate = 0.5
exponential_decay_steps = 200000
epoch = 2000
batch_size = 8

n_test_samples = 1
save_interval = 50

train_files = "./train_files.txt"
test_files = "./test_files.txt"
result_dir = "./result/"
load_path = None
