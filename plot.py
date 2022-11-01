import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(task, logs, n_seeds, l):
    log_folder = ['t' + str(task), 't' + str(task) + 'nBB', 't' + str(task) + 'nMQ']
    for label in log_folder:
        m_r = []
        for log, seeds in zip(logs, n_seeds):
            for i in range(seeds):
                df = pd.read_csv(log + label + '/' + str(i+1) + '/log.csv')
                m = list(df[[l]].to_numpy().transpose(1, 0)[0])
                m_r.append(sum(m)/len(m))
        std = np.std(m_r, ddof=1)
        mean = sum(m_r)/sum(n_seeds)
        print(l, 'rate for', label, 'is', mean, 'std', std)


def cal_test(task, logs, n_seeds, l):
    log_folder = ['t' + str(task), 't' + str(task) + 'nBB', 't' + str(task) + 'nMQ']#, 't' + str(task) + 'wBC']
    for label in log_folder:
        m = []
        for log, seeds in zip(logs, n_seeds):
            df = pd.read_csv(log + label + '/actor_performance.csv')
            m += (list(df[[l]].to_numpy().transpose(1, 0)[0][:seeds]))
        print(l, 'for', label, 'is', sum(m)/len(m), 'stde', np.std(m, ddof=1)/np.sqrt(len(m)))


def cal_episode(task, logs, n_seeds):
    log_folder = ['t' + str(task), 't' + str(task) + 'nBB', 't' + str(task) + 'nMQ', 't' + str(task) + 'wBC']
    for label in log_folder:
        e = 0
        for log, seeds in zip(logs, n_seeds):
            for i in range(seeds):
                df = pd.read_csv(log + label + '/' + str(i + 1) + '/log.csv')
                m = list(df[['episode']].to_numpy().transpose(1, 0)[0])
                e += len(m)
        e /= sum(n_seeds)
        print('n_episodes for', label, 'is', e)


def plot_test(log, seeds, polyak=0.01, train=False, task=1, ratio=False, loss=False, ensemble=False):
    time = set([])
    frame = []
    data = []
    for i in seeds:
        df = pd.read_csv(log + '/' + str(i) + '/log.csv')
        stamps = list(df[['frames']].to_numpy().transpose(1, 0)[0])
        frame.append(stamps)
        time = time | set(stamps)
        if train:
            rec = list(df[['return']].to_numpy().transpose(1, 0)[0])
            if log == 'saves_R_1_2/t1' or log =='saves_R_1_2/t2' or log == 'saves_R_1_2/t3':
                if task == 1:
                    rec[0] = 0.30
                elif task == 2:
                    rec[0] = 0.45
                else:
                    rec[0] = 0.14
            elif log == 'saves_2/t1wD' or log == 'saves_2/t2wD' or log == 'saves_2/t3wD' or log == 'save_w/t1' or log == 'save_w/t2' or log == 'save_w/t3':
                rec[0] = 0
            else:
                if task == 1:
                    rec[0] = 0.5766
                elif task == 2:
                    rec[0] = 0.8904
                else:
                    rec[0] = 0.2814
        elif loss:
            rec = list(df[['La']].to_numpy().transpose(1, 0)[0])
        elif ratio:
            rec = list(df[['ratio']].to_numpy().transpose(1, 0)[0])
        else:
            rec = list(df[['test']].to_numpy().transpose(1, 0)[0])
        data.append(rec)
    time = list(time)
    time.sort()
    for i in range(len(data)):
        for j in range(len(data[i])-1):
            data[i][j+1] = data[i][j] * (1-polyak) + data[i][j+1] * polyak
    for i in range(len(data)):
        count = 0
        data_new = []
        for time_step in time:
            if time_step < frame[i][count]:
                if count == 0:
                    if train or loss:
                        data_new.append(data[i][count])
                    else:
                        data_new.append(data[i][count] * time_step / frame[i][count])
                else:
                    data_new.append((data[i][count] - data[i][count-1]) * (time_step - frame[i][count-1])
                                    / (frame[i][count] - frame[i][count-1]) + data[i][count-1])
            elif time_step == frame[i][count]:
                data_new.append(data[i][count])
                if count < len(data[i]) - 1:
                    count += 1
            else:
                if count == len(data[i]) - 1:
                    data_new.append(data[i][count])
                else:
                    data_new.append((data[i][count+1] - data[i][count]) * (time_step - frame[i][count])
                                    / (frame[i][count+1] - frame[i][count]) + data[i][count])
                    if count < len(data[i]) - 1:
                        count += 1
        data[i] = data_new
    data = np.array(data)
    time = np.array(time)
    stderr = np.std(data, axis=0, ddof=1)  # / np.sqrt(data.shape[0])
    mean = np.mean(data, axis=0)
    time /= 1e3
    return time, mean, stderr


def plot_result(task, logs, n_seeds, labels, colors, polyak=0.01,
                train=False, ratio=False, loss=False, ensemble=False, image=False):
    mpl.style.use('seaborn')
    # color = ['royalblue', 'xkcd:green', 'xkcd:orange', 'crimson']
    for log, seeds, label, color in zip(logs, n_seeds, labels, colors):
        time, mean, stderr = plot_test(log=log, seeds=seeds, polyak=polyak,
                                       train=train, task=task, ratio=ratio,
                                       loss=loss, ensemble=ensemble)
        plt.plot(time, mean, label=label, c=color, lw=1.)
        plt.fill_between(time, mean - stderr, mean + stderr, color=color, alpha=0.15)
    baseline_data = {'train': [0.5766, 0.8904, 0.2814], 'test': [0.5766, 0.8904, 0.2814]}
    if train:
        plt.axhline(baseline_data['train'][task-1], xmin=0, xmax=1, c='gray', ls='--', label='Base Controller')
    if ensemble:
        max_x = 1000
    elif image:
        max_x = 200
    else:
        max_x = 200
    if not loss:
        plt.axis([0, max_x, 0, 1.0])
    plt.legend(prop={'size': 20})
    plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.17)
    plt.xlabel('Number of environment steps (x $\mathregular{10^3}$)', fontdict={'size': 25})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.ylabel('success rate', fontdict={'size': 25})
    plt.show()


if __name__ == '__main__':
     for t in range(3):
         plot_result(task=t+1,
                     logs=['saves/t'+str(t+1), 'saves/t'+str(t+1)+'nBB', 'saves/t'+str(t+1)+'nMQ',
                           'saves/t'+str(t+1)+'wBC', 'saves/t'+str(t+1)+'wD'],
                     n_seeds=[[1, 2, 3, 4, 5]] * 5,
                     labels=['WB', 'WBnBB', 'WBnMQ', 'WBwBC', 'WBwD'],
                     colors=['royalblue', 'xkcd:green', 'xkcd:orange', 'purple', 'crimson'],
                     polyak=0.01,
                     train=False, ratio=False, loss=False)
