import dplay_utils.xdeploy as xd
# Learning components
from games.aigym import AtariEnvironment_Pong
from experience_managers.preprocessors import GymAtariFramePreprocessor_Diff
from experience_managers.mem import ExperienceMemory
from networks.encoders.conv_encoders import DeepConvEncoder
from networks.decoders.policy_decoders import Decoder
from networks.nets import RLNet
from policies.sa_policies import Policy
from rl.train import OneStepPolicyGradientTrainer
from rl.keeppg import PolicyGradKeeper
from rl.keeppg import RLAlgorithm

FULL_PROJ_PATH = xd.get_project_path()
conf = xd.load_test_config(__file__, True)

# CREATE LEARNING COMPONENTS
env = AtariEnvironment_Pong()
preproc = GymAtariFramePreprocessor_Diff()
mem = ExperienceMemory(**conf['experience_opts'])
enc = DeepConvEncoder(conf['encoder_opts'])
conf['decoder_opts']['input_num'] = enc.get_feature_num({'height': preproc.im_height, 'width': preproc.im_width})
dec = Decoder(conf['decoder_opts'])
rlnet = RLNet(enc, dec)
policy = Policy(rlnet)
trainer = OneStepPolicyGradientTrainer(rlnet, mem, conf['trainer_opts'])
keeper = PolicyGradKeeper(objs=[enc, dec, policy, mem],  # objects has "save/load" interface
                          opts=conf['keeper_opts'])
alg = RLAlgorithm(keeper, env, preproc, mem, policy, trainer)
alg.run()