from src import resource_reader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# train_image = next(rr.request_data())[0]
# train_image = imresize(train_image, (100, 100))

for image, ground_truth,name in resource_reader.request_data():
    fig = plt.figure()
    a = fig.add_subplot(1, 6, 1,sharey=True)
    plt.imshow(image)
    a.set_title(name)
    a.axis('off')
    i = 2
    for segmentation, boundaries in ground_truth:
        a = fig.add_subplot(1, 6, i,sharey=True)
        imgplot = plt.imshow(boundaries)
        a.set_title('bound'+str(i-1))
        i = i+1
        a.axis('off')
    plt.show()
