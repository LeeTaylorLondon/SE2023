import gzip
import pickle


def store_obj(obj, fd):
    # Open a gzip file for writing
    with gzip.open(fd, 'wb') as f:
        # Use pickle to dump the list to the file
        pickle.dump(obj, f)
    # Mark EOF
    pass

def load_obj(fd):
    # Open a gzip file for reading
    with gzip.open(fd, 'rb') as f:
        # Use pickle to load the list from the file
        obj = pickle.load(f)
    # Return object to caller
    return obj


if __name__ == '__main__':
    # The list of objects to store
    my_list = [1, 2, 3, "hello", ["a", "b", "c"]]

    store_obj(my_list, "Object Storage/my_list.pkl.gz")

    obj = load_obj("Object Storage/my_list.pkl.gz")

    # Print the list
    print(obj)