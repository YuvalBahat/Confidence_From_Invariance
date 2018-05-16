import os
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

def SplitCifar10TestSet(dataset_folder,train_indicator=None):
    label_bytes = 1  # 2 for CIFAR-100
    height = 32
    width = 32
    depth = 3
    image_bytes = height * width * depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes
    read_counter,D_train_counter,D_test_counter = 0,0,0
    only_count = train_indicator is None
    if only_count:
        with open(os.path.join(dataset_folder, 'test_batch.bin'), 'rb') as f_read:
            while True:
                cur_sample = f_read.read(record_bytes)
                if len(cur_sample) == 0:
                    break
                read_counter += 1
    else:
        with open(os.path.join(dataset_folder,'test_batch.bin'), 'rb') as f_read, open(os.path.join(dataset_folder,'detector_train_batch.bin'), 'wb') as f_D_train,\
                open(os.path.join(dataset_folder,'detector_test_batch.bin'), 'wb') as f_D_test:
            while True:
                cur_sample = f_read.read(record_bytes)
                if len(cur_sample) == 0:
                    assert len(train_indicator)==read_counter,'Length of train_indicator mismatches the number of images in the dataset file'
                    break
                if train_indicator[read_counter]:
                    f_D_train.write(cur_sample)
                    D_train_counter += 1
                else:
                    f_D_test.write(cur_sample)
                    D_test_counter += 1
                read_counter += 1
        print('Read %d images, saved %d detector training images and %d detector validation images' % (read_counter, D_train_counter, D_test_counter))
    return read_counter

def ProcessValidationData(detector_logits,GT_detector_labels,MSR_values,figures_folder=None):
    detector_logits = np.concatenate(detector_logits)
    GT_detector_labels = np.concatenate(GT_detector_labels)
    MSR_values = np.concatenate(MSR_values)
    AUROC_CFI = metrics.roc_auc_score(y_true=GT_detector_labels, y_score=detector_logits)
    AUROC_MSR = metrics.roc_auc_score(y_true=GT_detector_labels, y_score=-1 * MSR_values)
    if figures_folder is not None:
        figure = plt.figure()
        if not os.path.isdir(figures_folder):
            os.mkdir(figures_folder)
        MSR_ROC = metrics.roc_curve(y_true=GT_detector_labels, y_score=-1 * MSR_values)
        plt.plot(MSR_ROC[0],MSR_ROC[1])
        CFI_ROC = metrics.roc_curve(y_true=GT_detector_labels, y_score=detector_logits)
        plt.plot(CFI_ROC[0],CFI_ROC[1])
        plt.legend(['Maximal softmax response (AUROC: %.3f)'%(AUROC_MSR),'Confidence from invariance (AUROC: %.3f)'%(AUROC_CFI)])
        plt.title('ROC curves on detector validation set')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig(os.path.join(figures_folder,'ROC_curves.png'))
        plt.close(figure)
    return AUROC_CFI,AUROC_MSR