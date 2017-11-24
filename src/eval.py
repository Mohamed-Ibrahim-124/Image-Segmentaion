import numpy as np 
from collections import Counter

def fmeasure(data, ground_truth):
    ELEM_NUM = 1
    PARTITION_VAL = 0
    gd_counter = Counter(
        ground_truth
    )
    assert type(data) == np.ndarray
    assert type(ground_truth) == np.ndarray
    f_measure = list()
    for cluster in np.unique(data):
        cluster_elements = ground_truth[ np.where(data == cluster) ]
        cluster_count = Counter(cluster_elements)
        max_partition = cluster_count.most_common(1)[0]
        purity = max_partition[ELEM_NUM] / cluster_elements.size
        recall = max_partition[ELEM_NUM] / gd_counter[max_partition[PARTITION_VAL]]
        f_measure.append(
            2 * purity * recall / (purity + recall)
        )
    assert len(f_measure) == np.size(np.unique(data))
    return sum(f_measure) / len(f_measure)

def conditional_entropy(data, ground_truth):
    total_entropy = 0
    for cluster in np.unique(data):
        cluster_elements = np.asarray(
            ground_truth[ np.where(data == cluster)]
        )
        elements_counter = Counter(cluster_elements)
        cluster_prob_list = list()
        cluster_entropy = 0
        for partition in np.unique(ground_truth):
            prob = elements_counter[partition] / np.size(cluster_elements)
            # print(prob)
            cluster_entropy -=  prob * np.log2(prob) if prob != 0 else 0
            assert np.isnan(prob) == False
            assert np.isnan(cluster_entropy)  == False
        total_entropy += cluster_entropy * cluster_elements.size 
    return total_entropy / data.size

if __name__ == "__main__":
    import resource_reader as rr 
    from kmeans import kmeans
    data_iter = rr.request_data()
    img, gt_iter = next(data_iter)
    for seg, bound in gt_iter:
        _,assignments,_ = kmeans(
            img.reshape(img.shape[0] * img.shape[1], 3),
            0.0001
        )
        print(
            conditional_entropy(
                assignments,
                rr.imresize(seg, rr.RES).flatten()
            )
        )
  