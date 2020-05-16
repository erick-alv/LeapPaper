from PIL import Image
import numpy as np
import argparse

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




def generate_images_array(l):
    return np.random.randint(low=0,high=256, size=(l, l, 3))




if __name__ == "__main__":
    #args = parse_args()
    #read_images(args)
    #ar_list = [generate_images_array(128) for _ in range(10)]
    #arm = np.array(ar_list, dtype=np.int32)
    #print(arm.ndim)
    #np.save('/home/erick/Pictures/testars.npy', arm)
    #image = Image.fromarray(ar, 'RGB')
    #image.show()
    arm = np.load('/home/erick/RL/LeapPaper/logdata/pnr/05-16-generate-vae-dataset-local/05-16-generate-vae-dataset-local_2020_05_16_15_55_17_id000--s17319/vae_dataset.npy',allow_pickle=True)
    print(arm.shape)
    print(arm.ndim)
