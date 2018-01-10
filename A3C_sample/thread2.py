from queue import Queue

import time

from threading import Thread
from threading import Lock
from threading import current_thread

print_lock = Lock()

def exampleJob(worker):
    time.sleep(1) # pretend to do some work.
    #with print_lock:
    print(current_thread().name,worker)


def do_staff(i,):

    while True:
        worker = q.get()
        exampleJob(worker)

        q.task_done()

q = Queue(maxsize=0)
num_threads = 5


for i in range(num_threads):
    worker = Thread(target=do_staff, args=(i,))
    worker.setDaemon(True)
    worker.start()


for x in range(10):
  q.put(x)

q.join()
