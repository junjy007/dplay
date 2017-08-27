import matplotlib.pyplot as plt
from matplotlib import gridspec

# State to image
def state_to_image(s, decay=0.5):
    sim = s[-1].astype(np.float)
    coeff = decay
    coeff_sum = 1.0
    for i in range(s.shape[0]-1, -1, -1):
        sim += s[i].astype(np.float) * coeff
        coeff_sum += coeff
        coeff *= decay
    sim /= coeff_sum
    sim = sim.astype(np.uint8)
    return sim

def draw_state_evaluation(state, action_val_pred, action_taken,
                          time_id, save_fname_prefix):
    num_actions = len(action_val_pred)
    im = state_to_image(state)
    plt.figure(1, figsize=(8, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[6, 1], width_ratios=[1,])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.cla()
    ax0.imshow(im, interpolation='nearest', cmap='gray')
    ax0.set_title("T={:d}".format(time_id))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax1.cla()
    ax1.set_xticks(np.arange(num_actions))
    ax1.set_xticklabels(['NOOP', 'UP', 'DOWN'])
    ax1.set_ylim([-1.0, +1])
    ax1.set_yticks(np.arange(-1, 1.01, .5))
    ax1.grid('on')
    ax1.bar(np.arange(num_actions), action_val_pred)
    if action_taken is not None:
        ax1.bar(action_taken, action_val_pred[action_taken], fc='r')
#     plt.tight_layout()
    fname = save_fname_prefix + '_{:05d}.png'.format(time_id)
    plt.savefig(fname)
    print "\rT={:d}".format(time_id),
    #plt.savefig('grid_figure.pdf')