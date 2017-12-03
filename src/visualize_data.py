import resource_reader as rr 
import matplotlib.pyplot as plt


def visualize_data(image, groundtruth, name):

        fig = plt.figure()
        a = fig.add_subplot(1, 6, 1)
        plt.imshow(image)
        a.set_title(name)
        a.axis('off')
        i = 2
        try:
            for segmentation, boundaries in groundtruth:
                a = fig.add_subplot(1, 6, i)
                plt.imshow(boundaries, cmap='Greys')
                a.axis('off')
                a = fig.add_subplot(2, 6, i)
                plt.imshow(segmentation)
                a.set_title('bound'+str(i-1))
                i = i+1
                a.axis('off')
        except TypeError as err:
            pass
            
            # for segmentation in groundtruth:
            #     a = fig.add_subplot(1, 6, i)
            #         plt.imshow(boundaries, cmap='Greys')
            #         a.axis('off')
            #         a = fig.add_subplot(2, 6, i)
            #         plt.imshow(segmentation)
            #         a.set_title('bound'+str(i-1))
            #         i = i+1
            #         a.axis('off')
        plt.show()
def show_images(images):
    fig, axis = plt.subplots(5)
    for img, ax in zip(images,axis):
        ax.imshow(img)
    fig.show()

if __name__ == '__main__':
    img, gtruth, s = next(rr.request_data())
    visualize_data(image=img, groundtruth=None, name=s)
