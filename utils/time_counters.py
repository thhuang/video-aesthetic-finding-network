from datetime import datetime

def time_counters(start_time=None, task_name='', print_time=False):
    if start_time == None:
        return datetime.now(), None
    else:
        dt = datetime.now() - start_time
        if print_time:
            print('{}: {}'.format(task_name, dt))
        return datetime.now(), dt