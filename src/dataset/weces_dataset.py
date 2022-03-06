from torch.utils.data import IterableDataset, get_worker_info
import math

from torch.utils.data.dataset import T_co


class WECESDataset(IterableDataset):
    def __init__(self, start, end):
        super(WECESDataset).__init__()
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))

    def __getitem__(self, index) -> T_co:
        pass
