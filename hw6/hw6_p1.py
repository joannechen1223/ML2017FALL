'''
https://wellecks.wordpress.com/tag/eigenfaces/
'''
import sys
from os import listdir
import numpy as np
from skimage import data, io


def load_image():
	image = []
	img_pack = listdir(sys.argv[1])
	for img in img_pack:
		new = io.imread('{file}/{img}'.format(file=sys.argv[1], img=img))
		image.append(new.flatten())
	image = np.asarray(image)
	image = image/255
	average = np.mean(image, axis=0)
	return image, average

def main():
	image, average = load_image()
	for i in range(415):
		image[i,:] = image[i,:]-average
	image = np.transpose(image)
	eigface, eigvalue, V = np.linalg.svd(image, full_matrices=False)
	#np.save('average.npy', average)
	#np.save('eigface.npy', eigface)
	img = io.imread('{file}/{img}'.format(file=sys.argv[1], img=sys.argv[2]))
	img = img.flatten()
	img = img/255
	img -= average
	weights = np.dot(img,eigface[:,:4])
	#print(np.shape(weights))
	recon = average + np.dot(weights, np.transpose(eigface[:,:4]))
	recon -= np.min(recon)
	recon /= np.max(recon)
	recon = np.reshape(recon, (600,600,3))
	io.imsave('reconstruction.jpg', recon)
	return


if __name__=="__main__":
	main()