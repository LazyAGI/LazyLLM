from lazyllm.launcher import Status

class ClientBase(object):
    def __init__(self, url):
        self.url = url

    def uniform_status(self, status):
        if status == 'Invalid':
            res = 'Invalid'
        elif status == 'Ready':
            res = 'Ready'
        elif Status[status] == Status.Done:
            res = 'Done'
        elif Status[status] == Status.Cancelled:
            res = 'Cancelled'
        elif Status[status] == Status.Failed:
            res = 'Failed'
        elif Status[status] == Status.Running:
            res = 'Running'
        else:  # TBSubmitted, InQueue, Pending
            res = 'Pending'
        return res
