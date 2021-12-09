import numpy as np
import matplotlib.pyplot as plt


def plot_loss(losses):
    x = [i for i in range(len(losses))]
    plt.plot(x, losses, color='#e49ab0')
    plt.title('Loss per epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show() 
    pass

def plot_acc(acc):
    x = [i for i in range(len(acc))]
    plt.plot(x, acc, color='#e49ab0')
    plt.title('Accuracy per epoch')
    plt.xlabel('Accuracy')
    plt.ylabel('Loss')
    plt.show() 
    pass


def plot_metrics(scores):

    score_names = ['Accuracy', 'F1', 'Precision', 'Recall']

    plt.figure(figsize=(15, 10), dpi=80)
    plt.bar(score_names, scores, width=0.4, color=['#eccfc3', '#957d95', '#904c77', '#e49ab0'], edgecolor="white")

    # plt.legend(['Mean', 'GRU add mean', 'GRU'], prop={"size":26}, loc=9, bbox_to_anchor=(0.5, -0.1),ncol=3)
    plt.ylabel("Score", size=30)
    plt.yticks(size=30)
    plt.xticks(size=30)
    # plt.ylim(0.4,1)
    # plt.xticks(x, cell_names, size=30)
    plt.title("Testing Performance", size=30)
    plt.show()
    pass

if __name__ == '__main__':
    loss_list = [6678.507038376548, 2870.2378761985087, 382.79791337793523, 599.8876093084162, 131.7403608668934, 124.14416174455123, 38.357993992892176, 15.704006975347346, 13.092526999386875, 10.386020920493387, 10.282957033677535, 12.075050180608576, 11.941755121404475, 12.06763796372847, 12.071805173700506, 11.130000677975742, 11.052288835698908, 9.911373051730068, 10.333863865245473, 10.305286234075373]
    
    acc_list = [0.13849431818181818, 0.1807528409090909, 0.23295454545454544, 0.16086647727272727, 0.23792613636363635, 0.2588778409090909, 0.26136363636363635, 0.18821022727272727, 0.26917613636363635, 0.33735795454545453, 0.3526278409090909, 0.2702414772727273, 0.2702414772727273, 0.2702414772727273, 0.26988636363636365, 0.2936789772727273, 0.30113636363636365, 0.3689630681818182, 0.3778409090909091, 0.37357954545454547]
    
    scores = [0.71, 0.37, 0.68, 0.54]
    
    # plot_loss(loss_list)

    # plot_acc(acc_list)

    plot_metrics(scores)



