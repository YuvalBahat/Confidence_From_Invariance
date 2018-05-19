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

def ProcessValidationData(detector_logits,GT_detector_labels,MSR_values,figures_folder=None,descriptor=None):
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
        CFI_ROC = metrics.roc_curve(y_true=GT_detector_labels, y_score=detector_logits)
        if descriptor is not None:
            np.savez(os.path.join(figures_folder,'%s.npz'%(descriptor)),CFI_ROC=CFI_ROC,AUROC_CFI=AUROC_CFI)
        additional_curves = [file for file in os.listdir(figures_folder) if '.npz' in file and file!='%s.npz'%(descriptor)]
        legend_list = []
        for file in additional_curves:
            curve = np.load(os.path.join(figures_folder,file))
            plt.plot(curve['CFI_ROC'][0],curve['CFI_ROC'][1])
            legend_list += ['CFI %s (AUROC: %.3f)'%(file[:-4],curve['AUROC_CFI'])]
        plt.plot(CFI_ROC[0],CFI_ROC[1])
        plt.plot(MSR_ROC[0],MSR_ROC[1])
        legend_list += ['Confidence from inv. %s (AUROC: %.3f)'%(descriptor,AUROC_CFI),'Maximal softmax response (AUROC: %.3f)'%(AUROC_MSR)]
        plt.legend(legend_list)
        plt.title('ROC curves on detector validation set')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig(os.path.join(figures_folder,'ROC_curves.png'))
        plt.close(figure)
    return AUROC_CFI,AUROC_MSR