import numpy as np
import struct
import os
import sys
import pickle

class KDTree:
    def __init__(self, points):
        self.tree = self.build_tree(points)
        
    def build_tree(self, points, depth=0):
        if len(points) == 0:
            return None
        
        axis = depth % points.shape[1]
        points = points[points[:, axis].argsort()]
        median_index = len(points) // 2

        return {
            'point': points[median_index],
            'left': self.build_tree(points[:median_index], depth + 1),
            'right': self.build_tree(points[median_index + 1:], depth + 1)
        }

    def query(self, point, k=1):
        return self._query(self.tree, point, k)

    def _query(self, node, point, k, depth=0):
        if node is None:
            return []

        axis = depth % point.shape[0]
        next_branch = None
        opposite_branch = None
        if point[axis] < node['point'][axis]:
            next_branch = node['left']
            opposite_branch = node['right']
        else:
            next_branch = node['right']
            opposite_branch = node['left']

        best_points = self._query(next_branch, point, k, depth + 1)
        best_points.append(node['point'])
        best_points = sorted(best_points, key=lambda x: np.linalg.norm(x - point))[:k]
        if len(best_points) < k or abs(point[axis] - node['point'][axis]) < np.linalg.norm(best_points[-1] - point):
            best_points += self._query(opposite_branch, point, k - len(best_points), depth + 1)

        return sorted(best_points, key=lambda x: np.linalg.norm(x - point))[:k]

def pca(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    X_reduced = X_centered.dot(selected_eigenvectors)
    
    return X_reduced

def fvecs_read_batch(file, dim, start, batch_size):
    file.seek(start * (dim + 1) * 4)
    buffer = file.read(batch_size * (dim + 1) * 4)
    if not buffer:
        raise IOError("Error reading batch from file.")
    vectors = []
    for i in range(batch_size):
        vector_dim = struct.unpack('i', buffer[i * (dim + 1) * 4: i * (dim + 1) * 4 + 4])[0]
        if vector_dim != dim:
            raise ValueError("Non-uniform vector sizes in file.")
        vector = struct.unpack('f' * dim, buffer[i * (dim + 1) * 4 + 4: (i + 1) * (dim + 1) * 4])
        vectors.append(vector)
    return np.array(vectors)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def remove_brackets(query):
    return query.replace('[', '').replace(']', '')

def main(query):
    query = remove_brackets(query)

    filename = "../datasets/gist/gist_base.fvecs" if len(query) > 3000 else "../datasets/sift/sift_base.fvecs"

    if not os.path.exists(filename):
        print(f"Error: Unable to open file: {filename}")
        return

    with open(filename, 'rb') as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0, os.SEEK_SET)

        if size == 0:
            print("Dataset is empty.")
            return

        dim = struct.unpack('i', file.read(4))[0]
        if dim <= 0:
            print(f"Invalid dimension in file: {filename}")
            return

        num_vectors = size // ((dim + 1) * 4)
        batch_size = 1000

        query_vector = [float(x) for x in query.split(',')]

        if len(query_vector) != dim:
            print("Query vector dimension mismatch.")
            return

        distances = []
        for start in range(0, num_vectors, batch_size):
            current_batch_size = min(batch_size, num_vectors - start)
            batch_vectors = fvecs_read_batch(file, dim, start, current_batch_size)

            for i in range(current_batch_size):
                dist = euclidean_distance(query_vector, batch_vectors[i])
                distances.append((dist, start + i))

        distances.sort(key=lambda x: x[0])

        result = ','.join(str(idx) for _, idx in distances[:10])
        print(result)

def read_fvecs(file_path):
    with open(file_path, 'rb') as f:
        while True:
            d = np.frombuffer(f.read(4), dtype=np.int32)
            if not d:
                break
            d = int(d[0])
            vec = np.frombuffer(f.read(4 * d), dtype=np.float32)
            yield vec

def process_dataset(data_path):
    data = np.array(list(read_fvecs(data_path)))
    transformer = pca()
    reduced_data = transformer.fit_transform(data)
    
    with open('reduced_data.pkl', 'wb') as f:
        pickle.dump(reduced_data, f)
    with open('transformer.pkl', 'wb') as f:
        pickle.dump(transformer, f)

def search_top10_nearest(query):
    with open('reduced_data.pkl', 'rb') as f:
        reduced_data = pickle.load(f)
    with open('transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    
    query_transformed = transformer.transform(np.array([query]))

    tree = KDTree(reduced_data)
    dists, inds = tree.query(query_transformed, k=10)
    print(','.join(map(str, inds)))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_py.py <command> <file_path_or_query>")
        sys.exit()
    
    command = sys.argv[1]
    if command == 'process':
        data_path = sys.argv[2]
        process_dataset(data_path)
    elif command == 'search':
        query = np.fromstring(sys.argv[2][1:-1], sep=',', dtype=np.float32)
        search_top10_nearest(query)