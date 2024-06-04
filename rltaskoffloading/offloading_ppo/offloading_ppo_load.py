# from offloading_ppo.offloading_ppo import S2SModel
# from offloading_ppo.offloading_ppo import Runner
# from environment.offloading_env import Resources
# from environment.offloading_env import OffloadingEnvironment


from rltaskoffloading.offloading_ppo.offloading_ppo import S2SModel
from rltaskoffloading.offloading_ppo.offloading_ppo import Runner
from rltaskoffloading.environment.offloading_env import Resources
from rltaskoffloading.environment.offloading_env import OffloadingEnvironment
import os 

import tensorflow as tf
import numpy as np

gamma=0.99
lam=0.95
ent_coef=0.01
vf_coef=0.5
max_grad_norm=0.5
#the following needs to comment 
#load_path = "./checkpoint/model.ckpt"
#load_path = None
load_path = "checkpoint/model.ckpt"


unit_type="layer_norm_lstm"
num_units=256
learning_rate=0.00005
supervised_learning_rate=0.00005
n_features=2
time_major=False 
is_attention=True
forget_bias=1.0
dropout=0
num_gpus=1
num_layers=2 
num_residual_layers=0 
is_greedy=False 
encode_dependencies = True
inference_model="sample" 
start_token=0
end_token=5
is_bidencoder=True

hparams = tf.contrib.training.HParams(
        unit_type=unit_type,
        num_units=num_units,
        learning_rate=learning_rate,
        supervised_learning_rate=supervised_learning_rate,
        n_features=n_features,
        time_major=time_major,
        is_attention=is_attention,
        forget_bias=forget_bias,
        dropout=dropout,
        num_gpus=num_gpus,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        is_greedy=is_greedy,
        inference_model=inference_model,
        start_token=start_token,
        end_token=end_token,
        is_bidencoder=is_bidencoder
    )





resource_cluster = Resources(mec_process_capable=(10.0*1024*1024),
                                 mobile_process_capable=(1.0*1024*1024),  bandwith_up=3.0, bandwith_dl=3.0)

env = OffloadingEnvironment(resource_cluster = resource_cluster,
                           batch_size=100,
                           graph_number=100,
                           #graph_file_paths=["../offloading_data/offload_random15_test/random.15."], #original 
                           graph_file_paths=["rltaskoffloading/offloading_data/offload_random15_test/random.15."],
                           time_major=False)

ob = tf.placeholder(dtype=tf.float32, shape=[None, None, env.input_dim])
ob_length = tf.placeholder(dtype=tf.int32, shape=[None])

make_model = lambda: S2SModel(hparams=hparams,ob=ob, ob_length=ob_length, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
model = make_model()
# #the following needs to comment 
# make_model = lambda: S2SModel(hparams=hparams,ob=ob, ob_length=ob_length, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
# model = make_model()


# eval_runner = Runner(env=env, model=model, nepisode=1, gamma=gamma, lam=lam)

# #original
# #Tc, Ec = eval_runner.sample_eval()
# Tc, Ec, additional_value = eval_runner.sample_eval()
# print(Tc)
# print(Ec)
# print(additional_value)
# running_cost, energy_consumption, running_qoe= eval_runner.sample_eval()
# print('the running cost is: ', running_cost)
# print('the energy consumption is:',energy_consumption)
# print('the running qoe is: ',running_qoe)

# greedy_Tc, greedy_Ec = eval_runner.greedy_eval()

# greedy_Tc = np.mean(greedy_Tc)
# greedy_Ec = np.mean(greedy_Ec)

# print("greedy run time cost: ", greedy_Tc)
# print("greedy energy consumption: ", greedy_Ec)


with tf.Session() as sess:
    # Create the model
    model = S2SModel(hparams=hparams, ob=ob, ob_length=ob_length, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    # Create a saver object
    saver = tf.train.Saver()
    
    # Check if the checkpoint file exists
    if os.path.exists(load_path + ".index"):
        # Restore the model from the checkpoint file
        saver.restore(sess, load_path)
        print("Model restored successfully from:", load_path)
    else:
        print("Error: Checkpoint file not found -", load_path)

    # Now 'model' contains the loaded model, and you can proceed with further steps
    eval_runner = Runner(env=env, model=model, nepisode=1, gamma=gamma, lam=lam)

    Tc, Ec, additional_value = eval_runner.sample_eval()
    print(Tc)
    print(Ec)
    print(additional_value)

