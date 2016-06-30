"""Library of L1 mock action tasks.

Postprocessing tasks are classes that:
    1) Can be initialized with only keyword arguments, and
    2) Are callable with the signature `task_instance(dediserser, ibeam,
    event_list)`. *event_list* is a list of postprocess.Event objects.
    3) Return None.

Feel free to write your own and register them in the INDEX.

"""

INDEX = {
        }

