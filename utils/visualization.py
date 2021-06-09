import os
import numpy as np
import matplotlib
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_training_results(results_filepath, output_path):

    val_losses = []
    train_losses = []
    val_aucs = []
    train_aucs = []
    val_center_errors = []
    train_center_errors = []

    with open(results_filepath, 'r') as results_file:
        file_lines = results_file.readlines()
        for line in file_lines:
            linedata = line.split(' ; ')
            linedata = [data.split()[1] for data in linedata]
            val_losses.append(float(linedata[5]))
            train_losses.append(float(linedata[2]))
            val_aucs.append(float(linedata[3]))
            train_aucs.append(float(linedata[0]))
            val_center_errors.append(float(linedata[4]))
            train_center_errors.append(float(linedata[1]))

    epochs = range(len(val_losses))

    # for line_idx, line in enumerate(file_lines):
    #     if 'Training results:' in line:
    #         lineidx_to_read_from = line_idx + 1
    #         break
    
    # for line_idx in range(lineidx_to_read_from, len(file_lines)):
    #     linedata = file_lines[line_idx].split()
        
    plt.figure(figsize=(10, 10))
    plt.xticks(epochs)

    plt.subplot(311)
    plt.plot(epochs, train_losses, 'orange', epochs, val_losses, 'b')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(['train_loss', 'val_loss'])

    plt.subplot(312)
    plt.plot(epochs, train_aucs, 'orange', epochs, val_aucs, 'b')
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train_auc', 'val_auc'])

    plt.subplot(313)
    plt.plot(epochs, train_center_errors, 'orange', epochs, val_center_errors, 'b')
    plt.ylabel('Center Error')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.legend(['train', 'val'])

    plt.savefig(os.path.join(output_path, 'training_results.png'))


def show_frame(frame, bbox, fig_n, pause=2):
    plt.ion()
    plt.clf()
    fig = plt.figure(fig_n)
    ax = fig.gca()
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    fig.show()
    fig.canvas.draw()
    plt.pause(pause)


def show_frame_and_response_map(frame, bbox, fig_n, crop_x, score, pause=2):
    fig = plt.figure(fig_n)
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    plt.ion()
    plt.show()
    plt.pause(pause)
    plt.clf()


def save_frame_and_response_map(frame, bbox, fig_n, crop_x, score, writer, fig):
    # fig = plt.figure(fig_n)
    plt.clf()
    ax = fig.add_subplot(131)
    ax.set_title('Tracked sequence')
    r = patches.Rectangle((bbox[0],bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    ax2 = fig.add_subplot(132)
    ax2.set_title('Context region')
    ax2.imshow(np.uint8(crop_x))
    ax2.spines['left'].set_position('center')
    ax2.spines['right'].set_color('none')
    ax2.spines['bottom'].set_position('center')
    ax2.spines['top'].set_color('none')
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax3 = fig.add_subplot(133)
    ax3.set_title('Response map')
    ax3.spines['left'].set_position('center')
    ax3.spines['right'].set_color('none')
    ax3.spines['bottom'].set_position('center')
    ax3.spines['top'].set_color('none')
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax3.imshow(np.uint8(score))

    # ax3.grid()
    writer.grab_frame()


def show_crops(crops, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(np.uint8(crops[0,:,:,:]))
    ax2.imshow(np.uint8(crops[1,:,:,:]))
    ax3.imshow(np.uint8(crops[2,:,:,:]))
    plt.ion()
    plt.show()
    plt.pause(0.001)


def show_scores(scores, fig_n):
    fig = plt.figure(fig_n)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.imshow(scores[0,:,:], interpolation='none', cmap='hot')
    ax2.imshow(scores[1,:,:], interpolation='none', cmap='hot')
    ax3.imshow(scores[2,:,:], interpolation='none', cmap='hot')
    plt.ion()
    plt.show()
    plt.pause(0.001)