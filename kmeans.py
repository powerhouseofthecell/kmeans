import numpy as np
import matplotlib.pyplot as plt

import os

from tqdm import tqdm

# * load_data : directory -> numpy array containing our images
def load_data(data_dir):
    img_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    imgs = np.array([plt.imread(f) for f in img_files])

    return imgs

# * preprocess : n x h x w x c numpy array -> n x h * w * c numpy array, shape of data, possibly normalized
def preprocess(data):
    n, h, w, c = data.shape

    flattened = np.reshape(data, (n, h * w * c))

    # # * normalize our data
    # img_mean = np.mean(flattened, axis=0)
    # img_std = np.std(flattened, axis=0)

    # # ! semi-normalized
    # normalized = (flattened - img_mean) / img_std

    new_data = flattened

    return new_data, h, w, c

class KMeansClassifier():
    def __init__(self, k=10, height=28, width=28, channel=1):
        self.k = k
        self.mean_images = None
        self.img_shape = (height, width, channel)

        self.avg_distances = list()

    def __str__(self):
        return f"KMeansClassifier Object: K={self.k}"

    # * takes in two numpy arrays, and returns euclidean distance
    def distance(self, img1, img2):
        deltas = img1 - img2
        squares = deltas ** 2
        sums = np.sum(squares)
        distance = np.sqrt(sums)

        return distance 

    def fit(self, data, iters=1):
        # * random init for the first clusters
        np.random.shuffle(data)
        clusters = np.split(data, self.k)

        self.mean_images = [np.mean(cluster, axis=0) for cluster in clusters]

        # ! iterations, don't hardcode 2
        for i in tqdm(range(iters)):
            # * initialize new clusters
            new_clusters = [list() for _ in range(self.k)]
            
            # * iterate over the data, and find which cluster each img is closest to, then assign to that cluster
            sum_min_distances = 0
            for img in tqdm(data):
                distances = [self.distance(img, cluster_mean) for cluster_mean in self.mean_images]

                index = np.argmin(distances)
                sum_min_distances += np.min(distances)

                new_clusters[index].append(img)
            
            self.avg_distances.append(sum_min_distances / len(data))
            
            self.mean_images = [np.mean(cluster, axis=0) for cluster in new_clusters]

    def display_means(self):
        plt.figure()
        for k, img in enumerate(self.mean_images):
            reshaped = np.reshape(img, self.img_shape)

            plt.imshow(reshaped)
            plt.title(f'Mean Image for Cluster #{k + 1}')
            plt.show()

    def display_avgs(self):
        plt.figure()
        plt.plot(self.avg_distances)
        plt.title('Average Distances over Iterations')
        plt.xlabel('Iteration Number')
        plt.ylabel('Average Distance')
        plt.show()

if __name__ == '__main__':
    try:
        # * load_data : directory -> numpy array containing our images
        data = load_data('stream_all')

        # * preprocess : n x h x w x c numpy array -> n x h * w * c numpy array, possibly normalized
        processed_data, h, w, c = preprocess(data)

        # * KMeansClassifier : the classifier itself
        kmeans = KMeansClassifier(k=4, height=h, width=w, channel=c)

        # * .fit : numpy array of data -> None
        kmeans.fit(processed_data, iters=10)

        # * .display_means : None -> None, displays our mean images
        kmeans.display_means()

        # * .display_avgs : None -> None, displays the average distances over iterations
        kmeans.display_avgs()
    
    except KeyboardInterrupt:
        print('User aborted')

