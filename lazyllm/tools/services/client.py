from lazyllm.launcher import Status

class ClientBase(object):
    """Base client class for managing service connections and status conversions.

Args:
    url (str): URL of the service endpoint.

Attributes:
    url: URL of the service endpoint.
"""
    def __init__(self, url):
        self.url = url

    def uniform_status(self, status):
        """Standardize task status string.

Args:
    status (str): Original status string.

**Returns:**

- str: Standardized status string, possible values include:
    - 'Invalid': Invalid status
    - 'Ready': Ready status
    - 'Done': Completed status
    - 'Cancelled': Cancelled status
    - 'Failed': Failed status
    - 'Running': Running status
    - 'Pending': Pending status (includes TBSubmitted, InQueue, Pending)
"""
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
