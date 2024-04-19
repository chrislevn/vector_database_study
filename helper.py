from functools import wraps
import timeit
import requests
import pandas as pd
from io import StringIO
import psutil
import os

def timeit_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed_time = timeit.default_timer() - start_time
        print(f"Elapsed time for function {func.__name__}: {elapsed_time:.2f}s")
        return result
    return wrapper

def prepare_data():
    res = requests.get('https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt')
    data = pd.read_csv(StringIO(res.text), sep='\t')
    sentences = data['sentence_A'].tolist()
    sentence_b = data['sentence_B'].tolist()
    sentences.extend(sentence_b)  # merge them
    # remove duplicates and NaN
    sentences = [word for word in list(set(sentences)) if type(word) is str]

    return sentences

# calculate CPU/GPU memory usage as wrapper function
def memory_usage(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024
        result = func(*args, **kwargs)
        end_memory = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage: {end_memory - start_memory:.2f} MB")
        return result
    return wrapper