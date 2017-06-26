"""
Making demo showing game play by trained AI. You need to provide
a game module <nm>, which contains a make_policy() factory method,
which in turn accepts checkpoint path and number ID, and returns a policy
object that I can use to play game, i.e. policy.get_action(state) returns
next action. Game module also contains a make_game() factory method,
which returns a game environment, behaving like OpenAI Gym environment.

Usage:
  demo.py mosaic game <gm> checkpoints <cp>... [--cpdir=<dir>] [--odir=<odir>]
  demo.py -h | --help

Options:
  -h --help      Show this screen.
  --cpdir=<dir>  Directory of checkpoints [default: checkpoints]
  --odir=<odir>  Output directory [default: demoout]
"""
import docopt
import importlib
import os
import numpy as np
from itertools import product
import cv2

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = np.c_[arr, 255*np.ones((len(arr),1), np.uint8)]
    return PIL.Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def record_game_play(env, pol):
    s = env.reset()
    states  = [s, ]
    rewards = [0., ]
    actions = []
    t = False
    while not t:
        a = pol(s)
        new_s, r, t, _ = env.step(a)
        s = new_s
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return states, actions, rewards

def make_mosaic(env, cpdir, cpids, outdir):
    all_states = {cp_:[] for cp_ in cpids}
    for i in cpids:
        pol = gm.make_policy(cpdir, i)
        all_states[i] = record_game_play(env, pol)[0]

    sample_frame = all_states[cpids[0]][0]  #type: np.ndarray
    frm_h, frm_w = sample_frame.shape[:2]
    chs = sample_frame.shape[2] if sample_frame.ndim > 2 else 1
    player_num = len(cpids)
    nr = nc = int(np.ceil(np.sqrt(player_num)))
    while nr * nc >= player_num + nc:
        nr -= 1
    moc_h, moc_w = int((frm_h + 2) * nr), int((frm_w + 2) * nc)

    # Make mosaic for each step of game play
    moc_im = np.zeros((moc_h, moc_w, chs), dtype=sample_frame.dtype)
    moc_grid = [(r * (frm_h + 2) + 1, (r + 1) * (frm_h + 2) - 1, 
    	         c * (frm_w + 2) + 1, (c + 1) * (frm_w + 2) - 1)
                for r, c in product(range(nr), range(nc))]

    frame_id = 0
    to_continue = True
    while to_continue:
        to_continue = False
        
        for cpid, g in zip(cpids, moc_grid):
            rl, rh, cl, ch = g
            if frame_id < len(all_states[cpid]):
                cpf_id = frame_id
                to_continue = True  # At least one checkpoint not finish
            else:
                cpf_id = -1  # if the checkpoint has been finished
            moc_im[rl:rh, cl:ch, ...] = all_states[cpid][cpf_id][...]

        if frame_id % 100 == 0:
        	print "\r Frame {}".format(frame_id),
            	# print "Prepare frame-{}, checkpoint-{}".format(frame_id, cpid)

        fname = os.path.join(outdir, 'f{:06d}.png'.format(frame_id))
        moc_im = moc_im[:, :, [2, 1, 0]]  # -> BGR for OpenCV to output
        cv2.imwrite(fname, moc_im)
        # v2.imshow("test", moc_im)
        # cv2.waitKey()
        frame_id += 1


if __name__ == '__main__':
    opts = docopt.docopt(__doc__)
    print opts

    if not os.path.exists(opts['--odir']):
        os.mkdir(opts['--odir'])

    # get the game-and-player module
    gmm_name, _ = os.path.splitext(opts['<gm>'])
    gm = importlib.import_module(gmm_name, '')

    env = gm.make_game()
    cpdir = opts['--cpdir']
    cpids = [int(id_) for id_ in opts['<cp>']]
    outdir = opts['--odir']

    if opts['mosaic']:
        make_mosaic(env, cpdir, cpids, outdir)


