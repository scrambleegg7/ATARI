import threading
import time
import logging

from queue import Queue

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

class myObject(object):

    def __init__(self, test=False):
        self.test = test
        self.counter = 0

    def increment(self):
        logging.debug("object increment ....")
        self.counter += 1

    #@property
    def getCounter(self):
        return self.counter


def consume(i,T_queue, myobject):

    T = T_queue.get()
    T_queue.put(T+1)
    t = 0

    #while(True):

    name = threading.currentThread().getName()

    while T < 15:
        logging.debug('Starting consume : time :%s Counter :%d ', time.strftime('%H:%M:%S') , T)
        myobject.increment()
        time.sleep(3)  # spend 3 seconds to process or consume the tiem

        t += 1
        T = T_queue.get()
        T_queue.put(T+1)

    T_queue.task_done()

    logging.debug('end : time :%s Counter :%d ', time.strftime('%H:%M:%S') , T)

    global done
    done = True


def producer(i,T_queue, myobject):
    # the main thread will put new items to the queue

    name = threading.currentThread().getName()
    #item = "item-" + str(i)

    T = T_queue.get()
    T_queue.put(T)
    logging.debug("starting producer ....%d ", T)

    while T < 15:
        T = T_queue.get()
        T_queue.put(T)
        obj_couner = myobject.getCounter()
        logging.debug("object counter %d", obj_couner)
        time.sleep(1)
    #T_queue.put(T+1)

    #T_queue.task_done()

    #q.join()

if __name__ == '__main__':


    done = False

    myobject = myObject()

    T_queue = Queue()
    T_queue.put(0)

    processes = []
    num_threads = 4
    for i in range(num_threads):
        processes.append(threading.Thread(target=consume, args=(i,T_queue, myobject) ) )

    processes.append(threading.Thread(target=producer, args=(i,T_queue, myobject) ) )

    for p in processes:
        p.setDaemon(True)
        p.start()

    while not done:
        time.sleep(.1)

    for p in processes:
        p.join()

    #T_queue.join()
