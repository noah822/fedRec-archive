from typing import Tuple
'''
    As the official doc of `flower` FL framework suggested in client_fn param specification:
    https://flower.dev/docs/apiref-flwr.html#flwr.simulation.start_simulation

    Any state required by the instance (model, dataset, hyperparameters, …)
    should be (re-)created in either the call to client_fn or the call to any of the client methods

'''

def _exec_method_callback(fn, self, param: Tuple):
    if fn is None:
        return 
    else:
        if param is None:
            return fn(self)
        else:
            return fn(self, *param)
        

def register_method_hook(
    enter: callable = None,
    exit: callable = None,
    enter_param: Tuple = None,
    exit_param: Tuple = None
):
    def _decorator(fn):
        '''
            every time call the method i.e _wrapper function
            wrap it as in a context manager
            - before enter:
              exec enter_callback
            - exec fn
            - after exit
              exec exit_callback 
        '''
        def _wrapper(self, *args, **kwargs):
            _exec_method_callback(
                enter, self, enter_param
            )
            res = fn(self, *args, **kwargs)
            _exec_method_callback(
                exit, self, exit_param
            )
            return res
        return _wrapper
    return _decorator
