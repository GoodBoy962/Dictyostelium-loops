def resize_image_arr(original_image, width, height):
    resize_image = np.zeros(shape=(width,height))
    for W in range(width):
        for H in range(height):
            new_width = int( W * original_image.shape[0] / width )
            new_height = int( H * original_image.shape[1] / height )
            resize_image[W][H] = original_image[new_width][new_height]
            
    return resize_image

 def plot_hic(arr, size=20):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    im = ax.matshow(arr, cmap='YlOrRd')
    fig.colorbar(im)