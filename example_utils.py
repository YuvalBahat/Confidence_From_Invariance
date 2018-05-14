import os

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
        return read_counter
    else:
        with open(os.path.join(dataset_folder,'test_batch.bin'), 'rb') as f_read, open(os.path.join(dataset_folder,'detector_train_batch.bin'), 'wb') as f_D_train,\
                open(os.path.join(dataset_folder,'detector_test_batch.bin'), 'wb') as f_D_test:
            while True:
                cur_sample = f_read.read(record_bytes)
                if len(cur_sample) == 0:
                    break
                if train_indicator[read_counter]:
                    f_D_train.write(cur_sample)
                    D_train_counter += 1
                else:
                    f_D_test.write(cur_sample)
                    D_test_counter += 1
                read_counter += 1
        print('Read %d images, saved %d detector training images and %d detector validation images' % (read_counter, D_train_counter, D_test_counter))
