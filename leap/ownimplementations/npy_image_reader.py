from PIL import Image
import numpy as np
import argparse
import os
import fnmatch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--npy_save_path', type=str)
    parser.add_argument('--npy_path', type=str)
    parser.add_argument('--image_save_path', type=str)
    return parser.parse_args()
def image_to_npy(image_path,npy_save_path, show=False):
    image = Image.open(image_path)
    image_array = np.array(image)
    np.save(npy_save_path, image_array)
    if show:
        print(image_array)

def npy_to_image(npy_path, image_save_path, display=False):
    def process_image(single_image_array, image_path):
        image = Image.fromarray(single_image_array, 'RGB')
        image.save(image_path)
        if display:
            image.show()

    npy_as_array = np.load(npy_path)
    if npy_as_array.ndim > 3:
        print('more than one image')
        max_times = min(10, npy_as_array.shape[0])
        name = None
        if image_save_path.endswith('.png'):
            name = image_save_path[:-4]
        else:
            name = image_save_path
        for i in range(max_times):
            process_image(npy_as_array[i], name+'_'+str(i)+'.png')
    else:
        process_image(npy_as_array, image_save_path)


def store_image_array_at(single_image_array, path_to_folder, img_name):
    '''
    :param dict:
    :param dict_key: key for the dict, where the images should be
    :return:
    '''
    '''files = [f for f in os.listdir(path_to_folder) if fnmatch.fnmatch(f, '*_id****.png')]
    print(files)'''
    if single_image_array.max() <= 1.0:
        #assume is normalized in [0,1]
        single_image_array *= 255.0

    image = Image.fromarray(single_image_array.astype(np.uint8), 'RGB')
    image.save(path_to_folder+img_name+'.png')

def make_random_imarray():
    return np.array([
        np.array([
            np.random.uniform(low=0, high=1, size=3)
            for _ in range(4)])
    for _ in range(4)])




if __name__ == "__main__":
    #args = parse_args()
    #read_images(args)
    #ar_list = [generate_images_array(128) for _ in range(10)]
    #arm = np.array(ar_list, dtype=np.int32)
    #print(arm.ndim)
    #np.save('/home/erick/Pictures/testars.npy', arm)
    #image = Image.fromarray(ar, 'RGB')
    #image.show()
    arm = np.load('/home/erick/log_leap/pnr/05-20-generate-vae-dataset-local/05-20-generate-vae-dataset-local_2020_05_20_22_58_16_id000--s94822/vae_dataset.npy',allow_pickle=True)
    print(arm.shape)
    print(arm.ndim)
