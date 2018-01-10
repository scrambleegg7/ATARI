import threading
import time
from queue   import Queue
def consume(q):
    while(True):
        name = threading.currentThread().getName()
        print("Thread: {0} start get item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')) )
        item = q.get();
        time.sleep(3)  # spend 3 seconds to process or consume the tiem
        print("Thread: {0} finish process item from queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')) )
        q.task_done()


def producer(q):
    # the main thread will put new items to the queue


    name = threading.currentThread().getName()
    print("Thread: {0} start put item into queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')) )
    item = "item-" + str(0)
    q.put(item)
    print("Thread: {0} successfully put item into queue[current size = {1}] at time = {2} \n".format(name, q.qsize(), time.strftime('%H:%M:%S')) )

    q.join()

if __name__ == '__main__':
    q = Queue(maxsize = 3)

    threads_num = 3  # three threads to consume
    for i in range(threads_num):
        t = threading.Thread(name = "ConsumerThread-"+str(i), target=consume, args=(q,))
        t.start()

    #1 thread to procuce
    t = threading.Thread(name = "ProducerThread", target=producer, args=(q,))
    t.start()

    q.join()
