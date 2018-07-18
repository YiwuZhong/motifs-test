# python log_parser.py detection or relation
import sys
import numpy as np
import matplotlib.pyplot as plt

mode = sys.argv[1]
if mode == 'detection':
    log = open("detector_loss.log")
    flag = False
    n = 0
    max_n = 5
    rpn_class_loss = []
    rpn_box_loss = []
    class_loss = []
    box_loss = []
    total = []

    for line in log:
        if 'overall' in line:
            flag = True
        elif flag is True:
            if n <max_n:
                n = n+1
                line = line.replace('\n', '')
                line = line.replace('\r', '')
                if n == 1:
                    rpn_class_loss.append(float(line.split(' ')[-1]))
                if n == 2:
                    rpn_box_loss.append(float(line.split(' ')[-1]))
                if n == 3:
                    class_loss.append(float(line.split(' ')[-1]))
                if n == 4:
                    box_loss.append(float(line.split(' ')[-1]))
                if n == 5:
                    total.append(float(line.split(' ')[-1]))
            else:
                n = 0
                flag = False
        else:
             continue

    plt.subplot(221)
    plt.scatter(np.arange(1,51,1), rpn_class_loss, 2, 'lightgreen')
    plt.plot(np.arange(1,51,1), rpn_class_loss,  'lightgreen', label = 'rpn_class_loss')
    plt.scatter(np.arange(1,51,1), class_loss, 2, 'darkgreen')
    plt.plot(np.arange(1,51,1), class_loss, 'darkgreen', label = 'final_class_loss')
    plt.title('Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line

    plt.subplot(222)
    plt.scatter(np.arange(1,51,1), rpn_box_loss, 2, 'lightblue')
    plt.plot(np.arange(1,51,1), rpn_box_loss, 'lightblue', label = 'rpn_box_loss')
    plt.scatter(np.arange(1,51,1), box_loss, 2, 'darkblue')
    plt.plot(np.arange(1,51,1), box_loss, 'darkblue', label = 'final_box_loss')
    plt.title('Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line

    plt.subplot(212)
    plt.scatter(np.arange(1,51,1), total, 2, 'orangered')
    plt.plot(np.arange(1,51,1), total, 'orangered', label = 'total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line

    plt.grid()
    #plt.xlim(0, 51)
    #plt.ylim(0, 1.1)
    plt.show()

elif mode == 'relation':
    log1 = open("sgcls_loss.log")
    log2 = open("sgdet_loss.log")
    flag = False
    n = 0
    max_n = 3
    cls_class_loss = []
    cls_rel_loss = []
    cls_total = []
    det_class_loss = []
    det_rel_loss = []
    det_total = []

    for line in log1:
        if 'overall' in line:
            flag = True
        elif flag is True:
            if n <max_n:
                n = n+1
                line = line.replace('\n', '')
                line = line.replace('\r', '')
                if n == 1:
                    cls_class_loss.append(float(line.split(' ')[-1]))
                if n == 2:
                    cls_rel_loss.append(float(line.split(' ')[-1]))
                if n == 3:
                    cls_total.append(float(line.split(' ')[-1]))
            else:
                n = 0
                flag = False
        else:
             continue

    for line in log2:
        if 'overall' in line:
            flag = True
        elif flag is True:
            if n <max_n:
                n = n+1
                line = line.replace('\n', '')
                line = line.replace('\r', '')
                if n == 1:
                    det_class_loss.append(float(line.split(' ')[-1]))
                if n == 2:
                    det_rel_loss.append(float(line.split(' ')[-1]))
                if n == 3:
                    det_total.append(float(line.split(' ')[-1]))
            else:
                n = 0
                flag = False
        else:
             continue

    plt.subplot(221)
    plt.scatter(np.arange(1,11,1), cls_class_loss, 2, 'lightgreen')
    plt.plot(np.arange(1,11,1), cls_class_loss,  'lightgreen', label = 'cls_class_loss')
    plt.scatter(np.arange(11,21,1), det_class_loss, 2, 'darkgreen')
    plt.plot(np.arange(11,21,1), det_class_loss, 'darkgreen', label = 'det_class_loss')
    plt.title('Class Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line
    plt.ylim(0, max(max(cls_class_loss, det_class_loss))*1.1)
    plt.grid()

    plt.subplot(222)
    plt.scatter(np.arange(1,11,1), cls_rel_loss, 2, 'lightgreen')
    plt.plot(np.arange(1,11,1), cls_rel_loss, 'lightblue', label ='cls_rel_loss')
    plt.scatter(np.arange(11,21,1), det_rel_loss, 2, 'darkgreen')
    plt.plot(np.arange(11,21,1), det_rel_loss, 'darkblue', label ='det_rel_loss')
    plt.title('Relation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line
    plt.ylim(0, max(max(cls_rel_loss, det_rel_loss)) * 1.1)
    plt.grid()

    plt.subplot(212)
    plt.scatter(np.arange(1,11,1), cls_total, 2, 'b')
    plt.plot(np.arange(1,11,1), cls_total, 'b', label = 'cls_total')
    plt.scatter(np.arange(11,21,1), det_total, 2, 'orangered')
    plt.plot(np.arange(11,21,1), det_total, 'orangered', label = 'det_total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()  # show the 'label' of line
    plt.ylim(0, max(max(cls_total, det_total))*1.1)
    plt.grid()

    plt.show()
else:
    ValueError('Wrong mode input')
